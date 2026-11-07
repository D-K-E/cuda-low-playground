// One MLP layer: relu(x * w + b), in linalg on tensors.
// x: 64x128, w: 128x256, b: 256, result: 64x256.
#id   = affine_map<(i, j) -> (i, j)>
#row  = affine_map<(i, j) -> (j)>

func.func @mlp(%x: tensor<64x128xf32>, %w: tensor<128x256xf32>,
               %b: tensor<256xf32>) -> tensor<64x256xf32> {
  %zero = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<64x256xf32>

  // acc = 0; acc += x * w
  %acc = linalg.fill ins(%zero : f32) outs(%init : tensor<64x256xf32>) -> tensor<64x256xf32>
  %mm = linalg.matmul ins(%x, %w : tensor<64x128xf32>, tensor<128x256xf32>)
                      outs(%acc : tensor<64x256xf32>) -> tensor<64x256xf32>

  // biased[i, j] = mm[i, j] + b[j]
  %biased = linalg.generic {indexing_maps = [#id, #row, #id],
                            iterator_types = ["parallel", "parallel"]}
      ins(%mm, %b : tensor<64x256xf32>, tensor<256xf32>)
      outs(%init : tensor<64x256xf32>) {
    ^bb0(%m: f32, %bias: f32, %unused: f32):
      %s = arith.addf %m, %bias : f32
      linalg.yield %s : f32
  } -> tensor<64x256xf32>

  // out[i, j] = max(biased[i, j], 0)
  %out = linalg.generic {indexing_maps = [#id, #id],
                         iterator_types = ["parallel", "parallel"]}
      ins(%biased : tensor<64x256xf32>)
      outs(%init : tensor<64x256xf32>) {
    ^bb0(%v: f32, %unused: f32):
      %r = arith.maximumf %v, %zero : f32
      linalg.yield %r : f32
  } -> tensor<64x256xf32>

  return %out : tensor<64x256xf32>
}
