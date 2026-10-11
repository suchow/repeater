//! Prints one line per card in the given Markdown files: hash, kind, question or cloze text, answer.
use std::path::Path;

use repeater::card::CardContent;
use repeater::parser::cards_from_md;

fn main() -> anyhow::Result<()> {
    for arg in std::env::args().skip(1) {
        for card in cards_from_md(Path::new(&arg))? {
            let (kind, text, answer) = match &card.content {
                CardContent::Basic { question, answer } => {
                    ("basic", question.clone(), answer.clone())
                }
                CardContent::Cloze { text, .. } => ("cloze", text.clone(), String::new()),
            };
            let esc = |s: &str| {
                s.replace('\\', "\\\\")
                    .replace('\n', "\\n")
                    .replace('\t', "\\t")
            };
            println!(
                "{}\t{}\t{}\t{}",
                card.card_hash,
                kind,
                esc(&text),
                esc(&answer)
            );
        }
    }
    Ok(())
}
