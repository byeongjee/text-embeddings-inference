//! Serves a fixed embedding schedule in-process: what the router does up to
//! `Infer`, driven by concurrent clients instead of HTTP, so that the run
//! ends and its makespan can be measured.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use text_embeddings_backend::{Backend, DType, ModelType, Pool};
use text_embeddings_core::infer::Infer;
use text_embeddings_core::queue::Queue;
use text_embeddings_core::tokenization::Tokenization;
use tokenizers::{Tokenizer, TruncationDirection};

const WORDS: [&str; 24] = [
    "the", "of", "system", "state", "value", "process", "model", "data", "please", "explain",
    "how", "does", "work", "with", "compare", "between", "memory", "cache", "network", "time",
    "result", "number", "large", "small",
];

#[derive(Parser)]
#[command(about = "Text embedding server benchmark")]
struct Cli {
    /// File the embeddings are written to.
    sink_path: PathBuf,
    /// Directory holding model.safetensors, config.json and tokenizer.json.
    #[arg(long)]
    model_dir: PathBuf,
    /// Concurrent clients, each sending its next request when its previous
    /// one is answered.
    #[arg(long, default_value_t = 16)]
    num_clients: usize,
    /// Requests each client sends.
    #[arg(long, default_value_t = 100)]
    requests_per_client: usize,
    /// Words per request.
    #[arg(long, default_value_t = 128)]
    input_words: usize,
    /// Seed the requests are generated from.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, value_enum, default_value_t = Pooling::Cls)]
    pooling: Pooling,
    /// Tokens a batch may hold.
    #[arg(long, default_value_t = 16384)]
    max_batch_tokens: usize,
}

#[derive(Clone, Copy, ValueEnum)]
enum Pooling {
    Cls,
    Mean,
    LastToken,
}

/// What the router reads from `config.json`.
#[derive(Deserialize)]
struct ModelConfig {
    model_type: String,
    max_position_embeddings: usize,
    pad_token_id: Option<usize>,
}

#[derive(Serialize)]
struct SinkRecord {
    client: usize,
    request: usize,
    embedding_head: [f32; 4],
    latency_ms: f64,
}

#[tokio::main]
async fn main() -> Result<()> {
    sweep::start_signal_trigger()?;
    let cli = Cli::parse();
    let start = Instant::now();

    let config: ModelConfig = serde_json::from_str(
        &fs::read_to_string(cli.model_dir.join("config.json")).context("config.json")?,
    )?;
    let mut tokenizer =
        Tokenizer::from_file(cli.model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
    tokenizer.with_padding(None);
    let position_offset = if ["xlm-roberta", "camembert", "roberta"].contains(&&*config.model_type)
    {
        config.pad_token_id.unwrap_or(0) + 1
    } else {
        0
    };
    let max_input_length =
        (config.max_position_embeddings - position_offset).min(cli.max_batch_tokens);
    let workers = (std::thread::available_parallelism()?.get() - 1).clamp(1, 64);
    let tokenization = Tokenization::new(
        workers,
        tokenizer,
        max_input_length,
        position_offset,
        None,
        None,
    );

    let pool = match cli.pooling {
        Pooling::Cls => Pool::Cls,
        Pooling::Mean => Pool::Mean,
        Pooling::LastToken => Pool::LastToken,
    };
    let backend = Backend::new(
        cli.model_dir.clone(),
        None,
        DType::default(),
        ModelType::Embedding(pool),
        None,
        "/tmp/tei-bench.sock".to_string(),
        None,
        "tei-bench".to_string(),
    )
    .await
    .context("create backend")?;
    backend.health().await.context("backend health")?;
    backend
        .warmup(
            max_input_length,
            cli.max_batch_tokens,
            backend.max_batch_size,
            backend.padded_model,
        )
        .await
        .context("warmup")?;
    let max_concurrent_requests = cli.num_clients;
    let queue = Queue::new(
        backend.padded_model,
        cli.max_batch_tokens,
        backend.max_batch_size,
        max_concurrent_requests,
    );
    let infer = Arc::new(Infer::new(
        tokenization,
        queue,
        max_concurrent_requests,
        backend,
    ));
    let loaded = start.elapsed();

    let serving = Instant::now();
    let clients: Vec<_> = (0..cli.num_clients)
        .map(|client| {
            let infer = Arc::clone(&infer);
            let mut rng = StdRng::seed_from_u64(cli.seed ^ ((client as u64) << 32));
            let requests = cli.requests_per_client;
            let words = cli.input_words;
            tokio::spawn(async move {
                let mut records = Vec::with_capacity(requests);
                for request in 0..requests {
                    let text = (0..words)
                        .map(|_| WORDS[rng.random_range(0..WORDS.len())])
                        .collect::<Vec<_>>()
                        .join(" ");
                    let permit = infer.acquire_permit().await;
                    let sent = Instant::now();
                    let response = infer
                        .embed_pooled(
                            text,
                            true,
                            TruncationDirection::Right,
                            None,
                            true,
                            None,
                            permit,
                        )
                        .await?;
                    let mut embedding_head = [0.0; 4];
                    embedding_head.copy_from_slice(&response.results[..4]);
                    records.push(SinkRecord {
                        client,
                        request,
                        embedding_head,
                        latency_ms: sent.elapsed().as_secs_f64() * 1e3,
                    });
                }
                anyhow::Ok(records)
            })
        })
        .collect();
    let mut records = Vec::new();
    for client in clients {
        records.extend(client.await??);
    }
    let served = serving.elapsed();

    let mut sink = BufWriter::new(File::create(&cli.sink_path)?);
    for record in &records {
        serde_json::to_writer(&mut sink, record)?;
        sink.write_all(b"\n")?;
    }
    sink.flush()?;

    let mut latencies: Vec<f64> = records.iter().map(|r| r.latency_ms).collect();
    latencies.sort_unstable_by(f64::total_cmp);
    let percentile = |q: f64| latencies[((latencies.len() - 1) as f64 * q).round() as usize];
    println!("requests         {}", records.len());
    println!("load s           {:.1}", loaded.as_secs_f64());
    println!("serve s          {:.1}", served.as_secs_f64());
    println!(
        "latency ms       p50 {:.0} p99 {:.0}",
        percentile(0.50),
        percentile(0.99)
    );
    println!(
        "weights bytes    {}",
        fs::metadata(cli.model_dir.join("model.safetensors"))?.len()
    );
    println!("sink             {}", cli.sink_path.display());
    Ok(())
}
