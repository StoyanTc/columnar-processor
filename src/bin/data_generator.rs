use rand::Rng;
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() {
    let path = "data/data_10m.csv";
    let file = File::create(path).unwrap();
    let mut writer = BufWriter::new(file);

    writeln!(writer, "id,value,category,region").unwrap();

    let mut rng = rand::rng();
    for i in 0..100_000_000 {
        let value = rng.random_range(1..1000);
        let category = ['A', 'B', 'C', 'D'][rng.random_range(0..4)];
        let region =
            ["US", "EU", "ASIA", "AFRICA", "AUSTRALIA", "SOUTH AMERICA"][rng.random_range(0..6)];
        writeln!(writer, "{},{},{},{}", i, value, category, region).unwrap();
    }

    println!("Sample CSV generated: {}", path);
}
