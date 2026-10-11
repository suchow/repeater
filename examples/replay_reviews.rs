//! Replays logged reviews against the database, in order, at their original times.
//!
//! Usage: replay_reviews <plan.tsv> <decks-root>
//! Each plan line is: file (relative to decks-root) TAB card hash TAB RFC 3339 time TAB pass|fail.
//! The database is the normal data dir, so set REPEATER_DATA_DIR to work on a copy.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use chrono::{DateTime, Utc};
use repeater::card::Card;
use repeater::crud::DB;
use repeater::fsrs::ReviewStatus;
use repeater::parser::cards_from_md;

const RETENTION: f32 = 0.9;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let (plan, root) = match args.as_slice() {
        [_, plan, root] => (PathBuf::from(plan), PathBuf::from(root)),
        _ => return Err(anyhow!("usage: replay_reviews <plan.tsv> <decks-root>")),
    };
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
        db.update_card_performance(card, status, Some(at), RETENTION)
            .await?;
        replayed += 1;
    }
    println!("replayed {replayed} reviews");
    Ok(())
}
