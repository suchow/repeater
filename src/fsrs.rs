use anyhow::{Context, Result};
use chrono::{DateTime, Duration, Utc};
use fsrs::{DEFAULT_PARAMETERS, FSRS, MemoryState};

const SECONDS_PER_DAY: f64 = 86_400.0;

pub const LEARN_AHEAD_THRESHOLD_MINS: Duration = Duration::minutes(20);

/// Same-day steps for a card you missed. A miss brings the card back after the first step, each
/// pass moves it to the next step, and passing the last step hands it back to FSRS. A miss at any
/// point starts the steps over. A card you pass without a miss skips them.
const LEARNING_STEPS: [Duration; 2] = [Duration::minutes(1), Duration::minutes(10)];

/// Shortest interval FSRS may set, so same-day reviews come only from the learning steps.
const MIN_FSRS_INTERVAL: Duration = Duration::days(1);

/// The learning step a card is waiting on, if any. Steps are the only intervals shorter than a
/// day, so a card's current interval identifies its step.
fn current_learning_step(perf: &ReviewedPerformance) -> Option<usize> {
    let interval_secs = (perf.interval_raw * SECONDS_PER_DAY).round() as i64;
    LEARNING_STEPS
        .iter()
        .position(|step| step.num_seconds() == interval_secs)
}

#[derive(Copy, Clone, PartialEq)]
pub enum ReviewStatus {
    Pass,
    Fail,
}

impl ReviewStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ReviewStatus::Pass => "Pass",
            ReviewStatus::Fail => "Fail",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReviewedPerformance {
    pub last_reviewed_at: DateTime<Utc>,
    pub stability: f64,
    pub difficulty: f64,
    pub interval_raw: f64,
    pub interval_days: usize,
    pub due_date: DateTime<Utc>,
    pub review_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Performance {
    #[default]
    New,
    Reviewed(ReviewedPerformance),
}

fn fsrs_model() -> Result<FSRS> {
    FSRS::new(Some(&DEFAULT_PARAMETERS)).context("failed to initialize FSRS model")
}

fn next_state_for_review(
    next_states: fsrs::NextStates,
    review_status: ReviewStatus,
) -> fsrs::ItemState {
    match review_status {
        ReviewStatus::Pass => next_states.good,
        ReviewStatus::Fail => next_states.again,
    }
}

pub fn update_performance(
    perf: Performance,
    review_status: ReviewStatus,
    reviewed_at: DateTime<Utc>,
    retention: f32,
) -> Result<ReviewedPerformance> {
    let (memory_state, last_reviewed_at, review_count) = match perf {
        Performance::New => (None, None, 0),
        Performance::Reviewed(ReviewedPerformance {
            last_reviewed_at,
            stability,
            difficulty,
            review_count,
            ..
        }) => (
            Some(MemoryState {
                stability: stability as f32,
                difficulty: difficulty as f32,
            }),
            Some(last_reviewed_at),
            review_count,
        ),
    };

    let learning_step = match &perf {
        Performance::Reviewed(p) => current_learning_step(p),
        Performance::New => None,
    };

    // Whole days since the last review. Same-day reviews (the learning steps) count as 0, so FSRS
    // uses its short-term formula for them.
    let elapsed_days = last_reviewed_at
        .map(|last| {
            let secs = reviewed_at.signed_duration_since(last).num_seconds().max(0) as f64;
            (secs / SECONDS_PER_DAY).round() as u32
        })
        .unwrap_or(0);

    let fsrs = fsrs_model()?;
    let next_states = fsrs.next_states(memory_state, retention, elapsed_days)?;
    let next_state = next_state_for_review(next_states, review_status);

    let interval_raw = next_state.interval as f64;
    let fsrs_seconds = (interval_raw * SECONDS_PER_DAY).round().max(1.0) as i64;
    let fsrs_duration = Duration::seconds(fsrs_seconds);

    let interval_duration = match (review_status, learning_step) {
        (ReviewStatus::Fail, _) => LEARNING_STEPS[0],
        (ReviewStatus::Pass, Some(step)) if step + 1 < LEARNING_STEPS.len() => {
            LEARNING_STEPS[step + 1]
        }
        (ReviewStatus::Pass, _) => fsrs_duration.max(MIN_FSRS_INTERVAL),
    };

    let interval_effective_days = interval_duration.num_seconds() as f64 / SECONDS_PER_DAY;
    let interval_days = interval_duration.num_days().max(0) as usize;
    let due_date = reviewed_at + interval_duration;

    Ok(ReviewedPerformance {
        last_reviewed_at: reviewed_at,
        stability: next_state.memory.stability as f64,
        difficulty: next_state.memory.difficulty as f64,
        interval_raw: interval_effective_days,
        interval_days,
        due_date,
        review_count: review_count + 1,
    })
}

/// FSRS's estimated probability of recalling the card at `at`, given its stability and the
/// actual time since its last review.
pub fn predicted_recall(perf: &ReviewedPerformance, at: DateTime<Utc>) -> f64 {
    let elapsed_ms = at
        .signed_duration_since(perf.last_reviewed_at)
        .num_milliseconds()
        .max(0) as f64;
    let elapsed_days = elapsed_ms / (SECONDS_PER_DAY * 1000.0);
    let memory = MemoryState {
        stability: perf.stability as f32,
        difficulty: perf.difficulty as f32,
    };
    // The last default parameter is FSRS-6's forgetting-curve decay.
    let decay = DEFAULT_PARAMETERS[DEFAULT_PARAMETERS.len() - 1];
    fsrs::current_retrievability(memory, elapsed_days as f32, decay) as f64
}

#[cfg(test)]
mod tests {
    use super::{
        Performance, ReviewStatus, ReviewedPerformance, predicted_recall, update_performance,
    };
    use chrono::{Duration, Utc};
    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-2
    }

    #[test]
    fn predicted_recall_starts_at_one_and_reaches_target_after_one_stability() {
        let t0 = Utc::now();
        let perf = ReviewedPerformance {
            last_reviewed_at: t0,
            stability: 2.0,
            difficulty: 5.0,
            interval_raw: 2.0,
            interval_days: 2,
            due_date: t0 + Duration::days(2),
            review_count: 3,
        };

        assert!((predicted_recall(&perf, t0) - 1.0).abs() < 1e-6);
        assert!((predicted_recall(&perf, t0 + Duration::days(2)) - 0.9).abs() < 1e-4);
        assert!(predicted_recall(&perf, t0 + Duration::days(10)) < 0.9);
    }

    #[test]
    fn first_pass_skips_learning_steps() {
        let reviewed_at = chrono::Utc::now();

        let result = update_performance(Performance::New, ReviewStatus::Pass, reviewed_at, 0.9);
        let ReviewedPerformance {
            last_reviewed_at,
            stability,
            difficulty,
            interval_raw,
            interval_days,
            due_date: _,
            review_count,
        } = result.unwrap();
        assert_eq!(last_reviewed_at, reviewed_at);
        assert!(approx_eq(stability, 2.30649995803833));
        assert!(approx_eq(difficulty, 2.1181039810180664));
        assert!(approx_eq(interval_raw, 2.3065046296296297));
        assert_eq!(interval_days, 2);
        assert_eq!(review_count, 1);
    }

    /// Reviews `perf` with each result in turn, each at the moment the card comes due.
    fn review_in_turn(mut perf: Performance, results: &[ReviewStatus]) -> Vec<ReviewedPerformance> {
        let mut reviewed_at = chrono::Utc::now();
        let mut out = Vec::new();
        for &result in results {
            let next = update_performance(perf, result, reviewed_at, 0.9).unwrap();
            reviewed_at = next.due_date;
            perf = Performance::Reviewed(next);
            out.push(next);
        }
        out
    }

    #[test]
    fn miss_goes_through_learning_steps() {
        use ReviewStatus::{Fail, Pass};
        let reviews = review_in_turn(Performance::New, &[Fail, Pass, Pass]);
        assert_eq!(
            reviews[0].due_date - reviews[0].last_reviewed_at,
            Duration::minutes(1)
        );
        assert_eq!(
            reviews[1].due_date - reviews[1].last_reviewed_at,
            Duration::minutes(10)
        );
        // Passing the last step hands the card to FSRS, which never schedules less than a day.
        assert_eq!(reviews[2].interval_raw, 1.0);
        assert_eq!(reviews[2].interval_days, 1);
    }

    #[test]
    fn miss_during_learning_steps_starts_them_over() {
        use ReviewStatus::{Fail, Pass};
        let reviews = review_in_turn(Performance::New, &[Fail, Pass, Fail, Pass]);
        assert_eq!(
            reviews[2].due_date - reviews[2].last_reviewed_at,
            Duration::minutes(1)
        );
        assert_eq!(
            reviews[3].due_date - reviews[3].last_reviewed_at,
            Duration::minutes(10)
        );
    }

    #[test]
    fn test_update_failed_review() {
        let now = chrono::Utc::now();
        let initial_perf = ReviewedPerformance {
            last_reviewed_at: now - Duration::days(4),
            stability: 3.0,
            difficulty: 5.0,
            interval_raw: 4.0,
            interval_days: 4,
            due_date: now + Duration::days(4),
            review_count: 3,
        };
        let result = update_performance(
            Performance::Reviewed(initial_perf),
            ReviewStatus::Fail,
            now,
            0.9,
        )
        .unwrap();
        // A lapse goes back into the learning steps.
        assert_eq!(result.due_date - now, Duration::minutes(1));
        assert!(result.stability < 3.0);
        assert_eq!(result.review_count, 4);
    }

    #[test]
    fn test_retention() {
        let now = chrono::Utc::now();
        let initial_perf = ReviewedPerformance {
            last_reviewed_at: now - Duration::days(4),
            stability: 3.0,
            difficulty: 5.0,
            interval_raw: 4.0,
            interval_days: 4,
            due_date: now,
            review_count: 4,
        };
        let interval_at = |retention| {
            update_performance(
                Performance::Reviewed(initial_perf),
                ReviewStatus::Pass,
                now,
                retention,
            )
            .unwrap()
            .interval_raw
        };
        assert_eq!(interval_at(0.9), 12.850474537037037);
        assert_eq!(interval_at(0.6), 346.84609953703705);
    }

    /// Test that cards can recover from a "stuck" state after multiple failures.
    #[test]
    fn stability_fix_after_failing() {
        let mut perf = Performance::default();
        let mut time_reviewed = chrono::Utc::now();

        // Simulate 4 consecutive failures - card enters "stuck" state
        for _ in 0..4 {
            let reviewed_perf =
                update_performance(perf, ReviewStatus::Fail, time_reviewed, 0.9).unwrap();
            time_reviewed = reviewed_perf.due_date;
            perf = Performance::Reviewed(reviewed_perf);
        }

        // After 4 fails: stability should be very low, difficulty very high
        let after_fails = match perf {
            Performance::Reviewed(p) => p,
            _ => panic!("expected Reviewed"),
        };
        assert!(
            after_fails.stability < 0.1,
            "After 4 fails, stability should be very low, got {}",
            after_fails.stability
        );
        assert!(
            after_fails.difficulty > 9.0,
            "After 4 fails, difficulty should be very high, got {}",
            after_fails.difficulty
        );

        // Simulate 5 consecutive passes - two clear the learning steps, then daily reviews
        // rebuild stability
        for _ in 0..5 {
            let reviewed_perf =
                update_performance(perf, ReviewStatus::Pass, time_reviewed, 0.9).unwrap();
            time_reviewed = reviewed_perf.due_date;
            perf = Performance::Reviewed(reviewed_perf);
        }

        let after_passes = match perf {
            Performance::Reviewed(p) => p,
            _ => panic!("expected Reviewed"),
        };
        assert!(
            after_passes.stability > 1.0,
            "After 5 passes, stability should recover above 1.0, got {}",
            after_passes.stability
        );
        assert!(
            after_passes.interval_raw > 1.0,
            "After 5 passes, interval should be > 1 day, got {}",
            after_passes.interval_raw
        );
    }
}
