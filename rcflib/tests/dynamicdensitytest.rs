extern crate rand;
extern crate rand_chacha;
extern crate rcflib;

use std::f32::consts::PI;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rcflib::common::multidimdatawithkey::MultiDimDataWithKey;
use rcflib::rcf::{RCFBuilder, RCFOptionsBuilder};

/// try cargo test --release
/// these tests are designed to be longish

#[test]
fn dynamic_density() {
    let base_dimension = 2;
    let shingle_size = 1;
    let number_of_trees = 50;
    let capacity = 256;
    let initial_accept_fraction = 0.1;
    let _point_store_capacity = capacity * number_of_trees + 1;
    let time_decay = 0.1 / capacity as f64;
    let bounding_box_cache_fraction = 1.0;
    let random_seed = 17;
    let parallel_enabled: bool = false;
    let store_attributes: bool = false;
    let internal_shingling: bool = false;
    let internal_rotation = false;

    let mut forest = RCFBuilder::<u64, u64>::new(base_dimension, shingle_size)
        .tree_capacity(capacity)
        .number_of_trees(number_of_trees)
        .random_seed(random_seed)
        .store_attributes(store_attributes)
        .parallel_enabled(parallel_enabled)
        .internal_shingling(internal_shingling)
        .internal_rotation(internal_rotation)
        .time_decay(time_decay)
        .initial_accept_fraction(initial_accept_fraction)
        .bounding_box_cache_fraction(bounding_box_cache_fraction)
        .build()
        .unwrap();

    let data: Vec<Vec<f32>> = generate_fan(1000, 3);
    let query_point = vec![0.7, 0.0f32];

    for degree in 0..360 {
        for j in 0..data.len() {
            forest
                .update(
                    &rotate_clockwise(&data[j], 2.0 * PI * degree as f32 / 360.0),
                    0,
                )
                .unwrap();
        }

        let density = forest.directional_density(&query_point).unwrap();
        let value = density.total();

        if (degree <= 60) || (120..=180).contains(&degree) || (240..=300).contains(&degree) {
            assert!(density.total() < 0.8 * capacity as f64); // the fan is above at 90,210,330
        }

        if (75..=105).contains(&degree)
            || (195..=225).contains(&degree)
            || (315..=345).contains(&degree)
        {
            assert!(density.total() > 0.5 * (capacity as f64));
        }

        // Testing for directionality
        // There can be unclear directionality when the
        // blades are right above

        let blade_above_in_y = density.low[1];
        let blade_below_in_y = density.high[1];
        let blade_to_the_left = density.high[0];
        let blade_to_the_right = density.low[0];

        // the tests below have a freedom of 10% of the total value
        if (75..=85).contains(&degree)
            || (195..=205).contains(&degree)
            || (315..=325).contains(&degree)
        {
            assert!(blade_above_in_y + 0.1 * value > blade_below_in_y);
            assert!(blade_above_in_y + 0.1 * value > blade_to_the_right);
        }

        if (95..=105).contains(&degree)
            || (215..=225).contains(&degree)
            || (335..=345).contains(&degree)
        {
            assert!(blade_below_in_y + 0.1 * value > blade_above_in_y);
            assert!(blade_below_in_y + 0.1 * value > blade_to_the_right);
        }

        if (60..=75).contains(&degree)
            || (180..=195).contains(&degree)
            || (300..=315).contains(&degree)
        {
            assert!(blade_above_in_y + 0.1 * value > blade_to_the_left);
            assert!(blade_above_in_y + 0.1 * value > blade_to_the_right);
        }

        if (105..=120).contains(&degree) || (225..=240).contains(&degree) || (degree >= 345) {
            assert!(blade_below_in_y + 0.1 * value > blade_to_the_left);
            assert!(blade_below_in_y + 0.1 * value > blade_to_the_right);
        }

        // fans are farthest to the left at 30,150 and 270
        if (15..=45).contains(&degree)
            || (135..=165).contains(&degree)
            || (255..=285).contains(&degree)
        {
            assert!(
                blade_to_the_left + 0.1 * value
                    > blade_above_in_y + blade_below_in_y + blade_to_the_right
            );
            assert!(blade_above_in_y + blade_below_in_y + 0.1 * value > blade_to_the_right);
        }
    }
}

fn generate_fan(num_per_blade: usize, blades: usize) -> Vec<Vec<f32>> {
    let mut data = Vec::new();

    let data_with_key = MultiDimDataWithKey::mixture(
        num_per_blade * blades,
        &[vec![0f32, 0f32]],
        &[vec![0.05, 0.2]],
        &[1.0f32],
        0,
    )
    .unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(72345);
    for point in data_with_key.data {
        let toss: f64 = rng.random();
        let mut i = 0;
        while i < blades + 1 {
            if toss < i as f64 / blades as f64 {
                let theta = 2.0 * PI * i as f32 / blades as f32;
                let mut vec = rotate_clockwise(&point, theta);
                vec[0] += 0.6 * theta.sin();
                vec[1] += 0.6 * theta.cos();
                data.push(vec);
                break;
            } else {
                i += 1;
            }
        }
    }
    data
}

fn rotate_clockwise(point: &[f32], theta: f32) -> Vec<f32> {
    let mut result = vec![0.0f32; 2];
    result[0] = theta.cos() * point[0] + theta.sin() * point[1];
    result[1] = -theta.sin() * point[0] + theta.cos() * point[1];
    result
}
