use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::card::{Card, CardContent};
use crate::cloze_utils::mask_cloze_text;
use crate::crud::DB;
use crate::fsrs::{LEARN_AHEAD_THRESHOLD_MINS, ReviewStatus};
use crate::llm::drill_preprocessor::{AIStatus, DrillPreprocessor};
use crate::notes::register_apple_notes_cards;
use crate::palette::Palette;
use crate::parser::register_all_cards;
use crate::parser::render_markdown;
use crate::parser::{Media, extract_media};
use crate::tui::Theme;
use crate::utils::{get_data_dir, pluralize};

use anyhow::{Context, Result, anyhow, bail};
use chrono::{DateTime, SecondsFormat, Utc};
use crossterm::event::{KeyEvent, KeyModifiers};
use crossterm::{
    event::{
        self, Event, KeyCode, KeyEventKind, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
        PushKeyboardEnhancementFlags,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Paragraph, Wrap},
};
use serde_json::{Value, json};
use tokio::sync::mpsc;

const MINUTES_PER_DAY: f64 = 24.0 * 60.0;
const FLASH_SECS: f64 = 2.0;

pub struct DrillOptions {
    pub paths: Vec<PathBuf>,
    pub card_limit: Option<usize>,
    pub new_card_limit: Option<usize>,
    pub rephrase_questions: bool,
    pub shuffle: bool,
    pub retention: f32,
    pub apple_notes: bool,
    /// Type each answer before revealing it, and append it to answers.jsonl in the data dir.
    pub typing: bool,
}

const ANSWER_LOG_FILE: &str = "answers.jsonl";

pub async fn run(db: &DB, opts: DrillOptions) -> Result<()> {
    validate_retention(opts.retention)?;
    let (hash_cards, _) = if opts.apple_notes {
        register_apple_notes_cards(db).await?
    } else {
        register_all_cards(db, opts.paths).await?
    };
    let mut cards_due_today = db
        .due_today(&hash_cards, opts.card_limit, opts.new_card_limit)
        .await?;

    if opts.shuffle {
        use rand::seq::SliceRandom;
        cards_due_today.shuffle(&mut rand::rng());
    }

    if cards_due_today.is_empty() {
        println!(
            "{}",
            Palette::paint(Palette::SUCCESS, "All caught up—no cards due today.")
        );
        return Ok(());
    }

    let answer_log = if opts.typing {
        Some(get_data_dir()?.join(ANSWER_LOG_FILE))
    } else {
        None
    };

    let drill_preprocessor =
        DrillPreprocessor::new(&cards_due_today, opts.rephrase_questions).await?;
    drill_preprocessor.initialize_card_status(&mut cards_due_today);
    start_drill_session(
        db,
        cards_due_today,
        drill_preprocessor,
        opts.retention,
        opts.shuffle,
        answer_log.clone(),
    )
    .await?;

    if let Some(log) = answer_log
        && log.exists()
    {
        println!("Typed answers are saved in {}", log.display());
    }

    Ok(())
}

fn validate_retention(retention: f32) -> Result<()> {
    if retention > 1.0 {
        bail!("Retention must be less than or equal to 1.0")
    }
    if retention < 0.65 {
        bail!("Retention must be greater than 0.65")
    }
    Ok(())
}
struct DrillState<'a> {
    db: &'a DB,
    cards: Vec<Card>,
    redo_cards: Vec<Card>,
    current_idx: usize,
    show_answer: bool,
    last_action: Option<LastAction>,
    current_medias: Vec<Media>,
    retention: f32,
    shuffle: bool,
    typed_answer: String,
    answer_log: Option<PathBuf>,
}
struct LastAction {
    action: ReviewStatus,
    show_again_duration: f64,
    last_reviewed_at: Instant,
}
impl LastAction {
    fn print(&self) -> String {
        let mut show_again = String::new();
        if self.show_again_duration <= 15.0 / MINUTES_PER_DAY {
            show_again.push_str("<15 mins");
        } else if self.show_again_duration <= 30.0 / MINUTES_PER_DAY {
            show_again.push_str("<30 mins");
        } else if self.show_again_duration <= 0.5 {
            show_again.push_str("<12 hours");
        } else if self.show_again_duration <= 1.0 {
            show_again.push_str("<1 day");
        } else {
            show_again.push_str(format!("{} days", self.show_again_duration as i64).as_str());
        }
        format!(" {} (See again in {})", self.action.label(), show_again,)
    }
}

impl<'a> DrillState<'a> {
    fn new(db: &'a DB, cards: Vec<Card>, retention: f32, shuffle: bool) -> Self {
        Self {
            db,
            cards,
            redo_cards: Vec::new(),
            current_idx: 0,
            show_answer: false,
            last_action: None,
            current_medias: Vec::new(),
            retention,
            shuffle,
            typed_answer: String::new(),
            answer_log: None,
        }
    }

    /// Turns on typing mode: answers are typed before reveal and logged to `answer_log`.
    fn with_typing(mut self, answer_log: PathBuf) -> Self {
        self.answer_log = Some(answer_log);
        self
    }

    fn typing(&self) -> bool {
        self.answer_log.is_some()
    }

    fn current_card(&mut self) -> Option<Card> {
        if self.current_idx >= self.cards.len() {
            if self.redo_cards.is_empty() {
                return None;
            }
            if self.shuffle {
                use rand::seq::SliceRandom;
                self.redo_cards.shuffle(&mut rand::rng());
            }
            self.cards = std::mem::take(&mut self.redo_cards);
            self.current_idx = 0;
        }
        self.cards.get(self.current_idx).cloned()
    }

    fn reveal_answer(&mut self) {
        self.show_answer = true;
    }

    async fn handle_review(&mut self, action: ReviewStatus) -> Result<()> {
        let current_card = self
            .current_card()
            .expect("card should exist when handling review");
        let show_again_duration = self
            .db
            .update_card_performance(&current_card, action, None, self.retention)
            .await?;
        if let Some(log) = &self.answer_log {
            let entry = answer_log_entry(&current_card, &self.typed_answer, action, Utc::now());
            append_answer_log(log, &entry)?;
        }
        self.typed_answer.clear();
        if action == ReviewStatus::Fail
            || show_again_duration
                < (LEARN_AHEAD_THRESHOLD_MINS.num_minutes() as f64 / MINUTES_PER_DAY)
        {
            self.redo_cards.push(current_card.clone());
        }

        self.last_action = Some(LastAction {
            action,
            show_again_duration,
            last_reviewed_at: std::time::Instant::now(),
        });
        self.current_idx += 1;
        self.show_answer = false;
        Ok(())
    }

    fn is_complete(&self) -> bool {
        self.current_idx >= self.cards.len() && self.redo_cards.is_empty()
    }

    fn apply_ai_update(&mut self, update: AiUpdate) {
        for card in self.cards.iter_mut().chain(self.redo_cards.iter_mut()) {
            if card.card_hash == update.card_hash {
                *card = update.card.clone();
                card.ai_status = AIStatus::AiEnhanced;
            }
        }
    }

    fn current_ai_pending(&self) -> bool {
        matches!(
            self.cards
                .get(self.current_idx)
                .map(|card| card.ai_status.clone()),
            Some(AIStatus::ClozeNeedDeletion | AIStatus::QuestionNeedRephrasing)
        )
    }
}

#[derive(Clone, Debug)]
struct AiUpdate {
    card_hash: String,
    card: Card,
}

async fn start_drill_session(
    db: &DB,
    cards: Vec<Card>,
    drill_preprocessor: DrillPreprocessor,
    retention: f32,
    shuffle: bool,
    answer_log: Option<PathBuf>,
) -> Result<()> {
    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(
        stdout,
        EnterAlternateScreen,
        // REPORT_ALTERNATE_KEYS makes shifted keys arrive as the shifted character
        // ('?' rather than '/' + Shift), which typed answers need.
        PushKeyboardEnhancementFlags(
            KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
                | KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
                | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
        )
    )
    .context("failed to configure terminal")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to start terminal")?;
    terminal.hide_cursor().context("failed to hide cursor")?;

    let (ai_updates_tx, mut ai_updates_rx) = mpsc::unbounded_channel();
    let mut ai_preprocess_handle = if drill_preprocessor.llm_required() {
        let ai_cards = cards.clone();
        Some(tokio::spawn(async move {
            preprocess_cards_in_order(drill_preprocessor, ai_cards, ai_updates_tx).await
        }))
    } else {
        None
    };

    let mut state = DrillState::new(db, cards, retention, shuffle);
    if let Some(log) = answer_log {
        state = state.with_typing(log);
    }

    let loop_result: Result<()> = async {
        loop {
            if state.is_complete() {
                break Ok(());
            }

            while let Ok(update) = ai_updates_rx.try_recv() {
                state.apply_ai_update(update);
            }

            if let Some(handle) = &mut ai_preprocess_handle
                && handle.is_finished()
            {
                let result = handle
                    .await
                    .map_err(|err| anyhow!("AI preprocessing task failed: {err}"))?;
                if let Err(err) = result {
                    break Err(err);
                }
                ai_preprocess_handle = None;
            }

            terminal
                .draw(|frame| {
                    let card = state
                        .current_card()
                        .expect("card should exist while session is active");
                    let area = frame.area();
                    frame.render_widget(Theme::backdrop(), area);
                    let chunks = Layout::default()
                        .direction(Direction::Vertical)
                        .constraints([Constraint::Min(5), Constraint::Length(5)])
                        .split(area);

                    let mut header_vec = vec![
                        Theme::label_span(format!(
                            "Card {}/{}",
                            state.current_idx + 1,
                            state.cards.len()
                        )),
                        Theme::bullet(),
                        Theme::span(format!("{} coming again", state.redo_cards.len())),
                        Theme::bullet(),
                        Theme::span(card.file_path.display().to_string()),
                    ];
                    if card.ai_status == AIStatus::AiEnhanced {
                        header_vec.push(Theme::bullet());
                        header_vec.push(Theme::key_chip("AI enhanced"));
                    }
                    let header_line = Line::from(header_vec);

                    let ai_pending = state.current_ai_pending();
                    let content = if ai_pending {
                        "Enhancing this card with AI...\n\nPlease wait.".to_string()
                    } else {
                        format_card_text(&card, state.show_answer)
                    };
                    let markdown = if state.typing() && !ai_pending {
                        card_view(&card, state.show_answer, &state.typed_answer)
                    } else {
                        render_markdown(&content)
                    };
                    state.current_medias = extract_media(&content, card.file_path.parent());

                    let card_widget = Paragraph::new(markdown)
                        .block(Theme::panel_with_line(header_line))
                        .wrap(Wrap { trim: false });
                    frame.render_widget(card_widget, chunks[0]);

                    let instructions = instructions_text(&state);
                    let footer = Paragraph::new(instructions)
                        .block(Theme::panel_with_line(Theme::section_header("Controls")));
                    frame.render_widget(footer, chunks[1]);
                })
                .context("failed to render frame")?;

            if event::poll(Duration::from_millis(16))?
                && let Event::Key(key) = event::read()?
            {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                if key.code == KeyCode::Esc
                    || (key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL))
                {
                    break Ok(());
                }
                let ai_pending = state.current_ai_pending();
                if state.typing() && !state.show_answer {
                    if !ai_pending {
                        match apply_typing_key(&mut state.typed_answer, key) {
                            TypingAction::Submit => state.reveal_answer(),
                            TypingAction::OpenMedia if !state.current_medias.is_empty() => {
                                state.current_medias[0].play()?;
                            }
                            _ => {}
                        }
                    }
                    continue;
                }
                match key.code {
                    KeyCode::Char(' ') | KeyCode::Enter if !ai_pending => {
                        if !state.show_answer {
                            state.reveal_answer();
                        } else {
                            state.handle_review(ReviewStatus::Pass).await?;
                        }
                    }
                    KeyCode::Char('F') | KeyCode::Char('f') if state.show_answer && !ai_pending => {
                        state.handle_review(ReviewStatus::Fail).await?;
                    }
                    KeyCode::Char('O') | KeyCode::Char('o')
                        if !ai_pending
                            && !state.show_answer
                            && !state.current_medias.is_empty() =>
                    {
                        state.current_medias[0].play()?;
                    }

                    _ => {}
                }
            }
        }
    }
    .await;

    teardown_terminal(&mut terminal)?;

    loop_result
}

fn teardown_terminal(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) -> Result<()> {
    disable_raw_mode().context("failed to disable raw mode")?;
    execute!(
        terminal.backend_mut(),
        PopKeyboardEnhancementFlags,
        LeaveAlternateScreen
    )
    .context("failed to restore terminal")?;
    terminal.show_cursor().context("failed to show cursor")?;
    Ok(())
}

fn instructions_text(state: &DrillState<'_>) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    if state.current_ai_pending() {
        lines.push(Line::from(vec![
            Theme::span("Enhancing card with AI"),
            Theme::bullet(),
            Theme::key_chip("Esc"),
            Theme::span(" / "),
            Theme::key_chip("Ctrl+C"),
            Theme::span(" exit"),
        ]));
    } else if state.show_answer {
        lines.push(Line::from(vec![
            Theme::key_chip("Space"),
            Theme::span(" or "),
            Theme::key_chip("Enter"),
            Span::styled(" Pass", Theme::success()),
            Theme::bullet(),
            Theme::key_chip("F"),
            Span::styled(" Fail", Theme::danger()),
            Theme::bullet(),
            Theme::key_chip("Esc"),
            Theme::span(" / "),
            Theme::key_chip("Ctrl+C"),
            Theme::span(" exit"),
        ]));
    } else if state.typing() {
        let mut line = vec![
            Theme::span("Type your answer"),
            Theme::bullet(),
            Theme::key_chip("Enter"),
            Theme::span(" show answer"),
            Theme::bullet(),
            Theme::key_chip("Esc"),
            Theme::span(" / "),
            Theme::key_chip("Ctrl+C"),
            Theme::span(" exit"),
        ];
        if !state.current_medias.is_empty() {
            let num_media = state.current_medias.len();
            line.push(Theme::bullet());
            line.push(Theme::span(format!(
                "{} found in card ",
                pluralize("media file", num_media)
            )));
            line.push(Theme::key_chip("Ctrl+O"));
            line.push(Theme::span(" open"));
        }
        lines.push(Line::from(line));
    } else {
        let mut line = vec![
            Theme::key_chip("Space"),
            Theme::span(" or "),
            Theme::key_chip("Enter"),
            Theme::span(" show answer"),
            Theme::bullet(),
            Theme::key_chip("Esc"),
            Theme::span(" / "),
            Theme::key_chip("Ctrl+C"),
            Theme::span(" exit"),
        ];
        if !state.current_medias.is_empty() {
            let num_media = state.current_medias.len();
            line.push(Theme::bullet());
            line.push(Theme::span(format!(
                "{} found in card ",
                pluralize("media file", num_media)
            )));
            line.push(Theme::key_chip("O"));
            line.push(Theme::span(" open"));
        }
        lines.push(Line::from(line));
    }

    if let Some(action) = &state.last_action
        && action.last_reviewed_at.elapsed().as_secs_f64() < FLASH_SECS
    {
        let style = match action.action {
            ReviewStatus::Pass => Theme::success(),
            ReviewStatus::Fail => Theme::danger(),
        };
        lines.push(Line::from(vec![
            Theme::span("Last:"),
            Span::styled(action.print(), style),
        ]));
    }

    lines
}

fn format_card_text(card: &Card, show_answer: bool) -> String {
    match &card.content {
        CardContent::Basic { question, answer } => {
            let mut text = format!("Q:\n{}\n\nA:\n", question);
            if show_answer {
                text.push_str(answer);
            }
            text
        }
        CardContent::Cloze { text, cloze_range } => {
            let body = match (cloze_range, show_answer) {
                (Some(range), false) => mask_cloze_text(text, range),
                _ => text.clone(),
            };
            format!("C:\n{}", body)
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum TypingAction {
    Edited,
    Submit,
    OpenMedia,
    Ignored,
}

/// Applies one key press to the typed answer. Letters that are shortcuts elsewhere
/// (Space, F, O) are typed as text here; Ctrl+O opens media instead.
fn apply_typing_key(typed: &mut String, key: KeyEvent) -> TypingAction {
    match key.code {
        KeyCode::Enter => TypingAction::Submit,
        KeyCode::Backspace => {
            typed.pop();
            TypingAction::Edited
        }
        KeyCode::Char('o') | KeyCode::Char('O')
            if key.modifiers.contains(KeyModifiers::CONTROL) =>
        {
            TypingAction::OpenMedia
        }
        KeyCode::Char(c)
            if !key
                .modifiers
                .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
        {
            // Terminals using the kitty keyboard protocol without alternate keys report
            // Shift+a as 'a' with SHIFT, so restore the uppercase letter ourselves.
            if key.modifiers.contains(KeyModifiers::SHIFT) {
                typed.push(c.to_ascii_uppercase());
            } else {
                typed.push(c);
            }
            TypingAction::Edited
        }
        _ => TypingAction::Ignored,
    }
}

/// Card body for typing mode: the typed answer sits on the "A:" line with a cursor
/// until the answer is revealed, then it's shown above the correct answer.
fn card_view(card: &Card, show_answer: bool, typed: &str) -> Text<'static> {
    let prompt = match &card.content {
        CardContent::Basic { question, .. } => format!("Q:\n{}", question),
        CardContent::Cloze { .. } => format_card_text(card, show_answer),
    };
    let mut view = render_markdown(&prompt);
    if show_answer {
        view.lines.push(Line::from(vec![
            Span::styled(
                "Your answer: ",
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::raw(typed.to_string()),
        ]));
        if let CardContent::Basic { answer, .. } = &card.content {
            view.lines.push(Line::default());
            view.lines
                .extend(render_markdown(&format!("A:\n{}", answer)).lines);
        }
    } else {
        view.lines.push(Line::from(vec![
            Span::raw("A: "),
            Span::raw(typed.to_string()),
            Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)),
        ]));
    }
    view
}

fn answer_log_entry(
    card: &Card,
    typed: &str,
    status: ReviewStatus,
    reviewed_at: DateTime<Utc>,
) -> Value {
    let (question, answer) = match &card.content {
        CardContent::Basic { question, answer } => (question.clone(), answer.clone()),
        CardContent::Cloze {
            text,
            cloze_range: Some(range),
        } => (
            mask_cloze_text(text, range),
            text[range.start..range.end]
                .trim_start_matches('[')
                .trim_end_matches(']')
                .to_string(),
        ),
        CardContent::Cloze {
            text,
            cloze_range: None,
        } => (text.clone(), String::new()),
    };
    json!({
        "reviewed_at": reviewed_at.to_rfc3339_opts(SecondsFormat::Secs, true),
        "file": card.file_path.display().to_string(),
        "card_hash": card.card_hash,
        "question": question,
        "answer": answer,
        "typed": typed,
        "result": status.label().to_lowercase(),
    })
}

fn append_answer_log(path: &Path, entry: &Value) -> Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open answer log {}", path.display()))?;
    writeln!(file, "{}", entry)
        .with_context(|| format!("failed to write answer log {}", path.display()))?;
    Ok(())
}

async fn preprocess_cards_in_order(
    drill_preprocessor: DrillPreprocessor,
    cards: Vec<Card>,
    updates: mpsc::UnboundedSender<AiUpdate>,
) -> Result<()> {
    for card in cards.into_iter() {
        let needs_ai = matches!(
            card.ai_status,
            AIStatus::ClozeNeedDeletion | AIStatus::QuestionNeedRephrasing
        );
        if !needs_ai {
            continue;
        }

        let mut updated_card = card.clone();
        drill_preprocessor
            .preprocess_cards(std::slice::from_mut(&mut updated_card))
            .await?;

        let _ = updates.send(AiUpdate {
            card_hash: updated_card.card_hash.clone(),
            card: updated_card,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::card::ClozeRange;
    use crate::cloze_utils::find_cloze_ranges;

    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;

    fn basic_card(question: &str, answer: &str) -> Card {
        basic_card_with_hash(question, answer, "hash")
    }

    fn basic_card_with_hash(question: &str, answer: &str, hash: &str) -> Card {
        let content = CardContent::Basic {
            question: question.into(),
            answer: answer.into(),
        };
        Card::new(PathBuf::from("test.md"), (0, 1), content, hash.into())
    }

    fn cloze_card(text: &str, blank_index: usize) -> Card {
        let (start, end) = find_cloze_ranges(text)[blank_index];
        Card::new(
            PathBuf::from("test.md"),
            (0, 1),
            CardContent::Cloze {
                text: text.into(),
                cloze_range: Some(ClozeRange::new(start, end).unwrap()),
            },
            "hash".into(),
        )
    }

    #[test]
    fn basic_card_hides_answer_until_revealed() {
        let card = basic_card("What?", "Answer");

        let hidden = format_card_text(&card, false);
        assert!(!hidden.contains("Answer"));

        let shown = format_card_text(&card, true);
        assert!(shown.contains("Answer"));
    }

    #[test]
    fn cloze_card_masks_until_answer_shown() {
        let card = cloze_card("Value [東京]", 0);

        let masked = format_card_text(&card, false);
        let placeholder = extract_placeholder(&masked);
        assert!(placeholder.chars().all(|c| c == '_'));
        assert!(placeholder.chars().count() >= 3);

        let revealed = format_card_text(&card, true);
        assert!(revealed.contains("[東京]"));
    }

    #[test]
    fn cloze_card_masks_second_blank_and_leaves_first_visible() {
        let card = cloze_card(
            "The [order] of a group is [the cardinality of its underlying set].",
            1,
        );

        let masked = format_card_text(&card, false);
        assert!(masked.contains("[order]"));
        assert!(!masked.contains("the cardinality of its underlying set"));

        let second_blank_start = masked.rfind('[').unwrap();
        let second_blank_end = masked[second_blank_start..].find(']').unwrap() + second_blank_start;
        let placeholder = &masked[second_blank_start + 1..second_blank_end];
        assert!(placeholder.chars().all(|c| c == '_'));

        let revealed = format_card_text(&card, true);
        assert!(revealed.contains("[order]"));
        assert!(revealed.contains("[the cardinality of its underlying set]"));
    }

    #[test]
    fn last_action_prints_human_friendly_intervals() {
        fn formatted(minutes: f64, status: ReviewStatus) -> String {
            let action = LastAction {
                action: status,
                show_again_duration: minutes / MINUTES_PER_DAY,
                last_reviewed_at: Instant::now(),
            };
            action.print()
        }

        assert_eq!(
            formatted(10.0, ReviewStatus::Pass),
            " Pass (See again in <15 mins)"
        );
        assert_eq!(
            formatted(20.0, ReviewStatus::Pass),
            " Pass (See again in <30 mins)"
        );
        assert_eq!(
            formatted(60.0, ReviewStatus::Pass),
            " Pass (See again in <12 hours)"
        );
        assert_eq!(
            formatted(22.0 * 60.0, ReviewStatus::Pass),
            " Pass (See again in <1 day)"
        );
        assert_eq!(
            formatted(3.0 * MINUTES_PER_DAY, ReviewStatus::Fail),
            " Fail (See again in 3 days)"
        );
    }

    #[test]
    fn instructions_show_answer_branch_includes_pass_and_fail() {
        let db = in_memory_db();
        let mut state = DrillState::new(&db, vec![basic_card("Q", "A")], 0.9, false);
        state.show_answer = true;

        let lines = instructions_text(&state);
        let commands = flatten_line(&lines[0]);

        assert!(commands.contains("Pass"));
        assert!(commands.contains("Fail"));
    }

    #[test]
    fn recent_last_action_is_displayed_in_instructions() {
        let db = in_memory_db();
        let mut state = DrillState::new(&db, vec![basic_card("Q", "A")], 0.9, false);
        state.show_answer = true;
        state.last_action = Some(LastAction {
            action: ReviewStatus::Fail,
            show_again_duration: 0.0,
            last_reviewed_at: Instant::now(),
        });

        let lines = instructions_text(&state);
        assert!(lines.len() >= 2);

        let last_line = flatten_line(lines.last().unwrap());
        assert!(last_line.contains("Last:"));
        assert!(last_line.contains("Fail"));
    }

    #[tokio::test]
    async fn redo_queue_keeps_insertion_order_when_shuffle_disabled() {
        let db = DB::new_in_memory().await.unwrap();
        let cards = vec![
            basic_card_with_hash("A?", "A", "hash-a"),
            basic_card_with_hash("B?", "B", "hash-b"),
            basic_card_with_hash("C?", "C", "hash-c"),
        ];
        db.add_cards_batch(&cards).await.unwrap();

        let mut state = DrillState::new(&db, cards.clone(), 0.9, false);
        for _ in 0..cards.len() {
            state.handle_review(ReviewStatus::Fail).await.unwrap();
        }

        let current = state.current_card().unwrap();
        assert_eq!(current.card_hash, "hash-a");

        let order = state
            .cards
            .iter()
            .map(|card| card.card_hash.as_str())
            .collect::<Vec<_>>();
        assert_eq!(order, vec!["hash-a", "hash-b", "hash-c"]);
        assert!(state.redo_cards.is_empty());
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn text_lines(text: &Text<'_>) -> Vec<String> {
        text.lines.iter().map(flatten_line).collect()
    }

    fn has_cursor(line: &Line<'_>) -> bool {
        line.spans
            .iter()
            .any(|span| span.style.add_modifier.contains(Modifier::REVERSED))
    }

    fn fixed_time() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-10-11T18:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    #[test]
    fn typing_keys_insert_letters_that_are_otherwise_shortcuts() {
        let mut typed = String::new();
        for c in ['f', ' ', 'o'] {
            assert_eq!(
                apply_typing_key(&mut typed, key(KeyCode::Char(c))),
                TypingAction::Edited
            );
        }
        assert_eq!(typed, "f o");
    }

    #[test]
    fn typing_backspace_removes_last_character() {
        let mut typed = String::from("東京");
        assert_eq!(
            apply_typing_key(&mut typed, key(KeyCode::Backspace)),
            TypingAction::Edited
        );
        assert_eq!(typed, "東");
    }

    #[test]
    fn typing_enter_submits_without_changing_text() {
        let mut typed = String::from("abc");
        assert_eq!(
            apply_typing_key(&mut typed, key(KeyCode::Enter)),
            TypingAction::Submit
        );
        assert_eq!(typed, "abc");
    }

    #[test]
    fn typing_ctrl_o_opens_media_instead_of_typing() {
        let mut typed = String::new();
        let ctrl_o = KeyEvent::new(KeyCode::Char('o'), KeyModifiers::CONTROL);
        assert_eq!(
            apply_typing_key(&mut typed, ctrl_o),
            TypingAction::OpenMedia
        );
        assert!(typed.is_empty());
    }

    #[test]
    fn typing_shift_letter_is_uppercase_even_without_alternate_keys() {
        let mut typed = String::new();
        apply_typing_key(
            &mut typed,
            KeyEvent::new(KeyCode::Char('a'), KeyModifiers::SHIFT),
        );
        apply_typing_key(
            &mut typed,
            KeyEvent::new(KeyCode::Char('B'), KeyModifiers::SHIFT),
        );
        assert_eq!(typed, "AB");
    }

    #[test]
    fn typing_view_puts_cursor_after_answer_label_and_hides_answer() {
        let card = basic_card("Capital of France?", "Paris");

        let view = card_view(&card, false, "Par");
        let lines = text_lines(&view);

        let last = view.lines.last().expect("view has lines");
        assert!(flatten_line(last).starts_with("A: Par"));
        assert!(has_cursor(last));
        assert!(lines.iter().any(|l| l.contains("Capital of France?")));
        assert!(!lines.iter().any(|l| l.contains("Paris")));
    }

    #[test]
    fn typing_view_shows_typed_text_literally_not_as_markdown() {
        let card = basic_card("Constructor name?", "__init__");

        let view = card_view(&card, false, "__init__ *x*");

        let last = flatten_line(view.lines.last().unwrap());
        assert!(last.starts_with("A: __init__ *x*"));
    }

    #[test]
    fn typing_view_after_reveal_shows_typed_and_correct_answer_without_cursor() {
        let card = basic_card("Capital of France?", "Paris");

        let view = card_view(&card, true, "Lyon");
        let lines = text_lines(&view);

        assert!(lines.iter().any(|l| l == "Your answer: Lyon"));
        assert!(lines.iter().any(|l| l.contains("Paris")));
        assert!(!view.lines.iter().any(has_cursor));
    }

    #[test]
    fn typing_view_for_cloze_masks_blank_and_adds_answer_line() {
        let card = cloze_card("The [order] of a group is [its size].", 1);

        let view = card_view(&card, false, "car");
        let lines = text_lines(&view);

        assert!(lines.iter().any(|l| l.contains("[order]")));
        assert!(!lines.iter().any(|l| l.contains("its size")));
        let last = view.lines.last().unwrap();
        assert!(flatten_line(last).starts_with("A: car"));
        assert!(has_cursor(last));
    }

    #[test]
    fn typing_view_for_revealed_cloze_shows_full_text_and_typed_answer() {
        let card = cloze_card("The [order] of a group is [its size].", 1);

        let lines = text_lines(&card_view(&card, true, "cardinality"));

        assert!(lines.iter().any(|l| l.contains("[its size]")));
        assert!(lines.iter().any(|l| l == "Your answer: cardinality"));
    }

    #[test]
    fn answer_log_entry_for_basic_card_records_all_fields() {
        let card = basic_card_with_hash("Capital of France?", "Paris", "hash-p");

        let entry = answer_log_entry(&card, "Lyon", ReviewStatus::Fail, fixed_time());

        assert_eq!(entry["reviewed_at"], "2026-10-11T18:00:00Z");
        assert_eq!(entry["file"], "test.md");
        assert_eq!(entry["card_hash"], "hash-p");
        assert_eq!(entry["question"], "Capital of France?");
        assert_eq!(entry["answer"], "Paris");
        assert_eq!(entry["typed"], "Lyon");
        assert_eq!(entry["result"], "fail");
    }

    #[test]
    fn answer_log_entry_for_cloze_uses_masked_text_and_hidden_words() {
        let card = cloze_card("The [order] of a group is [its size].", 1);

        let entry = answer_log_entry(&card, "its size", ReviewStatus::Pass, fixed_time());

        assert_eq!(entry["question"], "The [order] of a group is [________].");
        assert_eq!(entry["answer"], "its size");
        assert_eq!(entry["result"], "pass");
    }

    #[test]
    fn append_answer_log_writes_one_json_object_per_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("answers.jsonl");

        append_answer_log(&path, &serde_json::json!({"n": 1})).unwrap();
        append_answer_log(&path, &serde_json::json!({"n": 2})).unwrap();

        let contents = std::fs::read_to_string(&path).unwrap();
        let ns: Vec<i64> = contents
            .lines()
            .map(|line| {
                serde_json::from_str::<Value>(line).unwrap()["n"]
                    .as_i64()
                    .unwrap()
            })
            .collect();
        assert_eq!(ns, vec![1, 2]);
    }

    #[tokio::test]
    async fn typed_review_is_logged_and_input_is_cleared() {
        let db = DB::new_in_memory().await.unwrap();
        let cards = vec![basic_card_with_hash(
            "Capital of France?",
            "Paris",
            "hash-p",
        )];
        db.add_cards_batch(&cards).await.unwrap();
        let dir = tempfile::tempdir().unwrap();
        let log = dir.path().join("answers.jsonl");
        let mut state = DrillState::new(&db, cards, 0.9, false).with_typing(log.clone());
        state.typed_answer = "Lyon".into();
        state.reveal_answer();

        state.handle_review(ReviewStatus::Fail).await.unwrap();

        let contents = std::fs::read_to_string(&log).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 1);
        let entry: Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(entry["typed"], "Lyon");
        assert_eq!(entry["answer"], "Paris");
        assert_eq!(entry["result"], "fail");
        assert!(state.typed_answer.is_empty());
    }

    #[test]
    fn typing_mode_instructions_ask_for_typed_answer() {
        let db = in_memory_db();
        let dir = tempfile::tempdir().unwrap();
        let state = DrillState::new(&db, vec![basic_card("Q", "A")], 0.9, false)
            .with_typing(dir.path().join("answers.jsonl"));

        let line = flatten_line(&instructions_text(&state)[0]);

        assert!(line.contains("Type your answer"));
        assert!(line.contains("Enter"));
        assert!(!line.contains("Space"));
    }

    fn extract_placeholder(text: &str) -> String {
        let start = text.find('[').unwrap();
        let end = text[start..].find(']').unwrap() + start;
        text[start + 1..end].to_string()
    }

    fn flatten_line(line: &Line<'_>) -> String {
        line.spans
            .iter()
            .map(|span| span.content.to_string())
            .collect::<String>()
    }

    fn in_memory_db() -> DB {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(DB::new_in_memory())
            .unwrap()
    }
}
