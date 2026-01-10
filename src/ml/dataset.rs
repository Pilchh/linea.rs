use std::{fmt, fs};

use crate::{matrix::Matrix, ml::feature::Feature, vector::Vector};

#[derive(Debug)]
pub struct Dataset {
    features: Vec<Feature>,
    samples: Matrix,
}

impl Dataset {
    pub fn from_csv(path: &str) -> Dataset {
        // TODO: Check if file exists
        let contents = fs::read_to_string(path).expect("should have been able to read file");
        let rows: Vec<&str> = contents.lines().collect();

        // TODO: Check size of rows is greater than one
        let feature_row = rows[0].split(',').collect::<Vec<&str>>();
        let mut features: Vec<Feature> = feature_row
            .iter()
            .map(|&name| Feature {
                name: name.to_string(),
            })
            .collect();

        features.insert(
            0,
            Feature {
                name: "bias".into(),
            },
        );

        let sample_rows = &rows[1..].to_vec();
        let mut samples_vector = Vec::with_capacity(features.len() * sample_rows.len());

        for row in sample_rows {
            // Adding bias column
            samples_vector.push(1.0);
            let values = row.split(',').collect::<Vec<&str>>();

            for value in values {
                let float: f64 = value.parse().expect("not a valid float");
                samples_vector.push(float);
            }
        }

        let samples = Matrix::from_vec(sample_rows.len(), features.len(), samples_vector)
            .expect("dimensions should be correct");

        Dataset { features, samples }
    }

    pub fn samples(&self) -> Matrix {
        self.samples.clone()
    }

    pub fn features(&self) -> Vec<Feature> {
        self.features.clone()
    }
}

impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Print header (feature names)
        for (i, feature) in self.features.iter().enumerate() {
            if i < self.features.len() - 1 {
                write!(f, "{}, ", feature.name);
            } else {
                write!(f, "{}", feature.name);
            }
        }

        let total = if self.samples.rows > 10 {
            10 * self.samples.cols
        } else {
            self.samples.rows * self.samples.cols
        };

        for i in 0..total {
            if i % self.features.len() != 0 {
                write!(f, "{:.2}, ", self.samples.data[i]);
            } else {
                write!(f, "\n{:.2}, ", self.samples.data[i]);
            }
        }

        Ok(())
    }
}
