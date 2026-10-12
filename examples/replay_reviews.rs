//! Replays logged reviews against the database, in order, at their original times.
//!
//! Usage: replay_reviews <plan.tsv> <decks-root> [states.jsonl]
//! Each plan line is: file (relative to decks-root) TAB card hash TAB RFC 3339 time TAB pass|fail.
//! The database is the normal data dir, so set REPEATER_DATA_DIR to work on a copy.
//! With states.jsonl, it also writes one line per review with the scheduler's `before` and
//! `after` state and `predicted_recall`, in the same format as the drill's review log.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, SecondsFormat, Utc};
use repeater::card::Card;
use repeater::crud::DB;
use repeater::fsrs::{Performance, ReviewStatus, predicted_recall};
use repeater::parser::cards_from_md;
use serde_json::{Value, json};

const RETENTION: f32 = 0.9;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (plan, root, states_path) = match args.as_slice() {
        [_, plan, root] => (PathBuf::from(plan), PathBuf::from(root), None),
        [_, plan, root, states] => (
            PathBuf::from(plan),
            PathBuf::from(root),
            Some(PathBuf::from(states)),
        ),
        _ => {
            return Err(anyhow!(
                "usage: replay_reviews <plan.tsv> <decks-root> [states.jsonl]"
            ));
        }
    };
    let mut states = String::new();
    let db = DB::new().await?;
    let mut decks: HashMap<String, Vec<Card>> = HashMap::new();
    let mut replayed = 0;
    for line in std::fs::read_to_string(&plan)?.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        let [file, hash, at, result] = fields[..] else {
            return Err(anyhow!("bad plan line: {line}"));
        };
        if !decks.contains_key(file) {
            decks.insert(
                file.to_string(),
                cards_from_md(&root.join(Path::new(file)))?,
            );
        }
        let card = decks[file]
            .iter()
            .find(|c| c.card_hash == hash)
            .with_context(|| format!("no card {hash} in {file}"))?;
        let status = match result {
            "pass" => ReviewStatus::Pass,
            "fail" => ReviewStatus::Fail,
            other => return Err(anyhow!("bad result {other}")),
        };
        let at: DateTime<Utc> = DateTime::parse_from_rfc3339(at)?.with_timezone(&Utc);
        let before = db.get_card_performance(card).await?;
        db.update_card_performance(card, status, Some(at), RETENTION)
            .await?;
        let after = db.get_card_performance(card).await?;
        let recall = match before {
            Performance::Reviewed(p) => Some(predicted_recall(&p, at)),
            Performance::New => None,
        };
        let state = json!({
            "before": performance_json(&before),
            "after": performance_json(&after),
            "predicted_recall": recall,
        });
        states.push_str(&format!("{state}\n"));
        replayed += 1;
    }
    if let Some(path) = states_path {
        std::fs::write(path, states)?;
    }
    println!("replayed {replayed} reviews");
    Ok(())
}

fn timestamp(t: DateTime<Utc>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Millis, true)
}

/// Matches `performance_json` in the drill's review log.
fn performance_json(perf: &Performance) -> Value {
    match perf {
        Performance::New => Value::Null,
        Performance::Reviewed(p) => json!({
            "last_reviewed_at": timestamp(p.last_reviewed_at),
            "stability": p.stability,
            "difficulty": p.difficulty,
            "interval_days": p.interval_raw,
            "due_date": timestamp(p.due_date),
            "review_count": p.review_count,
        }),
    }
}
