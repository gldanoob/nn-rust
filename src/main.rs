mod linalg;
mod nn;

use csv;
use linalg::Matrix;

fn main() {
    mlp_demo().unwrap();
}

fn one_hot<const N: usize>(v: &str, t: [&str; N]) -> [f64; N] {
    if !t.contains(&v) {
        panic!("Invalid value: {}", v);
    }
    t.map(|x| if x == v { 1.0 } else { 0.0 })
}

fn mlp_demo() -> Result<(), Box<dyn std::error::Error>> {
    // Data science in Rust 💀

    let mut data = csv::Reader::from_path("data/data_heart_disease.csv").unwrap();
    let mut processed = Vec::<f64>::new();
    let mut targets = Vec::<f64>::new();

    for result in data.records() {
        let record = result?;

        // TODO: handle missing values
        if record.into_iter().any(|x| x.is_empty()) {
            continue;
        }

        let age = record[0].parse::<f64>()?;
        let [sex_m, sex_f] = one_hot(&record[1], ["M", "F"]);
        let [cp_ata, cp_asy, cp_nap, cp_ta] = one_hot(&record[2], ["ATA", "ASY", "NAP", "TA"]);
        let rbp = record[3].parse::<f64>()?;
        let chol = record[4].parse::<f64>()?;
        let fbs = record[5].parse::<f64>()?;
        let [recg_normal, recg_st, recg_lvh] = one_hot(&record[6], ["Normal", "ST", "LVH"]);
        let max_hr = record[7].parse::<f64>()?;
        let [exang_yes, exang_no] = one_hot(&record[8], ["Y", "N"]);
        let oldpeak = record[9].parse::<f64>()?;
        let [slope_flat, slope_up, slope_down] = one_hot(&record[10], ["Flat", "Up", "Down"]);
        let hr_gap = record[11].parse::<f64>()?;
        let target = record[12].parse::<f64>()?;

        processed.extend(&[
            age,
            sex_m,
            sex_f,
            cp_ata,
            cp_asy,
            cp_nap,
            cp_ta,
            rbp,
            chol,
            fbs,
            recg_normal,
            recg_st,
            recg_lvh,
            max_hr,
            exang_yes,
            exang_no,
            oldpeak,
            slope_flat,
            slope_up,
            slope_down,
            hr_gap,
        ]);

        targets.push(target);
    }

    let mut x = Matrix::from(processed.len() / 21, 21, &processed);
    let y = Matrix::from(targets.len(), 1, &targets);
    x = x.normalize();

    // 4/5 for training, 1/5 for testing
    let n = x.rows();
    let n0 = n * 4 / 5;

    let (train_x, train_y) = (x.slice_y(0..n0), y.slice_y(0..n0));
    let (test_x, test_y) = (x.slice_y(n0..n), y.slice_y(n0..n));

    let mut nn = nn::MLP::new(&[21, 10, 1]);
    println!("Training...");
    for i in 0..1000 {
        nn.train_batch(&train_x, &train_y, 0.01);
        if i % 100 == 0 {
            println!("Epoch: {}", i);
        }
    }

    println!();
    println!("Target  | Output");
    let mut true_pos = 0;
    let mut true_negs = 0;
    let positives = (0..n * 1 / 5)
        .filter(|i| test_y[(i.to_owned(), 0)] == 1.0)
        .count();

    for i in 0..n * 1 / 5 {
        let target = test_y.get_row(i)[(0, 0)];
        let output = nn.run(&test_x.get_row(i).transpose())[(0, 0)];
        println!("{:.5} | {:.5}", target, output);
        if target == 1.0 && output > 0.5 {
            true_pos += 1;
        } else if target == 0.0 && output < 0.5 {
            true_negs += 1;
        }
    }

    println!();

    println!(
        "Accuracy: {}",
        (true_negs + true_pos) as f64 / test_y.rows() as f64
    );
    println!("TP rate: {}", true_pos as f64 / positives as f64);
    println!(
        "TN rate: {}",
        true_negs as f64 / (test_y.rows() - positives) as f64
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{linalg::*, nn};
    #[test]
    fn test_nn() {
        // XOR
        let mut nn = nn::MLP::new(&[2, 5, 1]);
        for _ in 0..1000 {
            nn.train_batch(
                &Matrix::from(4, 2, &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
                &Matrix::from(4, 1, &[0.0, 1.0, 1.0, 0.0]),
                1.0,
            );
        }
        println!(
            "0 0: {:}",
            nn.run(&Matrix::from(1, 2, &[0.0, 0.0]).transpose())
        );
        println!(
            "0 1: {:}",
            nn.run(&Matrix::from(1, 2, &[0.0, 1.0]).transpose())
        );
        println!(
            "1 0: {:}",
            nn.run(&Matrix::from(1, 2, &[1.0, 0.0]).transpose())
        );
        println!(
            "1 1: {:}",
            nn.run(&Matrix::from(1, 2, &[1.0, 1.0]).transpose())
        );
    }

    // AI generated tests
    #[test]
    fn test_matrix() {
        let m = Matrix::new(2, 3);
        assert_eq!(m.size(), (2, 3));
        let m = Matrix::from(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(m.size(), (2, 3));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(0, 2)], 3.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 1)], 5.0);
        assert_eq!(m[(1, 2)], 6.0);

        let m = m.transpose();
        assert_eq!(m.size(), (3, 2));
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 4.0);
        assert_eq!(m[(1, 0)], 2.0);
        assert_eq!(m[(1, 1)], 5.0);
        assert_eq!(m[(2, 0)], 3.0);
        assert_eq!(m[(2, 1)], 6.0);

        let m = m.map(|x| x * 2.0);
        assert_eq!(m.size(), (3, 2));
        assert_eq!(m[(0, 0)], 2.0);
        assert_eq!(m[(0, 1)], 8.0);
        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 1)], 10.0);
        assert_eq!(m[(2, 0)], 6.0);

        let m = m.get_row(1);
        assert_eq!(m.size(), (1, 2));
        assert_eq!(m[(0, 0)], 4.0);
        assert_eq!(m[(0, 1)], 10.0);

        let m = Matrix::from(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let n = Matrix::from(2, 3, &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let res = m.add(&n);
        assert_eq!(res.size(), (2, 3));
        assert_eq!(res[(0, 0)], 6.0);
        assert_eq!(res[(0, 1)], 8.0);
        assert_eq!(res[(0, 2)], 10.0);
        assert_eq!(res[(1, 0)], 12.0);
        assert_eq!(res[(1, 1)], 14.0);
        assert_eq!(res[(1, 2)], 16.0);

        let res = m.dot(&n.transpose());
        assert_eq!(res.size(), (2, 2));
        assert_eq!(res[(0, 0)], 38.0);
        assert_eq!(res[(0, 1)], 56.0);
        assert_eq!(res[(1, 0)], 92.0);
        assert_eq!(res[(1, 1)], 137.0);

        let a = Matrix::from(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let res = a.normalize();
        println!("{:?}", res);
    }
}
