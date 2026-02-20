/*!
Performance benchmarks for medical image preprocessing operations.

Run with: cargo bench
*/

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ndarray::{Array2, Array3};
use med_core_rs::preprocessing::*;

fn generate_test_image(size: usize) -> Array2<f32> {
    Array2::from_shape_fn((size, size), |(i, j)| {
        ((i * j) as f32 / (size * size) as f32) * 255.0
    })
}

fn generate_test_batch(batch_size: usize, img_size: usize) -> Array3<f32> {
    Array3::from_shape_fn((batch_size, img_size, img_size), |(_, i, j)| {
        ((i * j) as f32 / (img_size * img_size) as f32) * 255.0
    })
}

fn bench_normalize_minmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_minmax");

    for size in [256, 512, 1024].iter() {
        let image = generate_test_image(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let img = black_box(&image);
                normalize_minmax(&img.view())
            });
        });
    }

    group.finish();
}

fn bench_normalize_percentile(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_percentile");

    for size in [256, 512, 1024].iter() {
        let image = generate_test_image(*size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let img = black_box(&image);
                normalize_percentile(&img.view(), 1.0, 99.0)
            });
        });
    }

    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_normalize");

    for batch_size in [10, 50, 100].iter() {
        let images = generate_test_batch(*batch_size, 512);

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let imgs = black_box(&images);
                    // Simulate batch normalization
                    (0..imgs.shape()[0])
                        .map(|i| {
                            let img = imgs.slice(ndarray::s![i, .., ..]).to_owned();
                            normalize_minmax(&img.view())
                        })
                        .collect::<Vec<_>>()
                });
            }
        );
    }

    group.finish();
}

fn bench_center_crop(c: &mut Criterion) {
    let mut group = c.benchmark_group("center_crop");

    let image = generate_test_image(1024);

    for target_size in [224, 512, 768].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(target_size),
            target_size,
            |b, &size| {
                b.iter(|| {
                    let img = black_box(&image);
                    center_crop(&img.view(), size, size)
                });
            }
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_normalize_minmax,
    bench_normalize_percentile,
    bench_batch_processing,
    bench_center_crop
);

criterion_main!(benches);
