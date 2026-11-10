mod audio;
mod audio8;
mod cli;
mod engine;
mod nemo_speech;
mod parent;
mod pull;
mod server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args_os().skip(1);
    if args.next().is_some_and(|arg| arg == nemo_speech::GUARD_ARG) {
        nemo_speech::run_guard(args);
    }
    // Logs go to stderr so stdout stays reserved for the ready handshake and
    // for command output.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    cli::run().await
}
