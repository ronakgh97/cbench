use clap::{Parser, Subcommand, arg};

#[derive(Parser)]
#[command(
    name = "cbench",
    version = "1.0.0-theta",
    about = "BLAS based cpu micro-benching tool"
)]
struct CliArgs {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Run {
        #[arg(short, long)]
        single: bool,

        #[arg(short, long)]
        all: bool,

        #[arg(short, long)]
        max_threads: Option<usize>,
    },
}

fn main() {
    let args = CliArgs::parse();

    match args.command {
        Some(Command::Run {
            single,
            all,
            max_threads,
        }) => {
            todo!()
        }
        None => {
            todo!()
        }
    }
}

fn print_ascii() {}
