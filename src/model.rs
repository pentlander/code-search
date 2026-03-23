//! NomicBert model for CodeRankEmbed (MLX Metal GPU).
//!
//! Architecture: 12-layer BERT with rotary embeddings, SwiGLU MLP, post-norm.
//! Config: 768 hidden, 12 heads, 64 head_dim, vocab 30528, rotary_base=1000.

use mlx_rs::{
    builder::Builder,
    error::Exception,
    module::Module,
    nn::{self, Embedding, LayerNorm, Linear, Rope, RopeInput},
    ops,
    Array,
};
use mlx_macros::ModuleParameters;
use mlx_rs::module::ModuleParametersExt;
use std::path::Path;
use tokenizers::Tokenizer;

const MAX_LENGTH: usize = 128;
const QUERY_PREFIX: &str = "Represent this query for searching relevant code: ";

/// Model configuration loaded from config.json.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct NomicBertConfig {
    pub vocab_size: i32,
    pub n_embd: i32,
    pub n_head: i32,
    pub n_layer: i32,
    #[serde(default = "default_n_inner")]
    pub n_inner: Option<i32>,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_epsilon: f32,
    #[serde(default = "default_rotary_base")]
    pub rotary_emb_base: f32,
    #[serde(default = "default_rotary_fraction")]
    pub rotary_emb_fraction: f32,
    #[serde(default)]
    pub rotary_emb_interleaved: bool,
    #[serde(default)]
    pub qkv_proj_bias: bool,
    #[serde(default)]
    pub mlp_fc1_bias: bool,
    #[serde(default)]
    pub mlp_fc2_bias: bool,
    #[serde(default)]
    pub prenorm: bool,
}

fn default_n_inner() -> Option<i32> { None }
fn default_layer_norm_eps() -> f32 { 1e-12 }
fn default_rotary_base() -> f32 { 10000.0 }
fn default_rotary_fraction() -> f32 { 1.0 }

// ── Model structs ──────────────────────────────────────────────────────

#[derive(Debug, ModuleParameters)]
pub struct NomicEmbeddings {
    #[param]
    pub word_embeddings: Embedding,
}

#[derive(Debug, ModuleParameters)]
#[allow(non_snake_case)]
pub struct NomicAttention {
    #[param]
    pub Wqkv: Linear,
    #[param]
    pub out_proj: Linear,
    rope: Rope,
    num_heads: i32,
    head_dim: i32,
}

#[derive(Debug, ModuleParameters)]
pub struct NomicMLP {
    #[param]
    pub fc11: Linear,
    #[param]
    pub fc12: Linear,
    #[param]
    pub fc2: Linear,
}

#[derive(Debug, ModuleParameters)]
pub struct NomicBlock {
    #[param]
    pub attn: NomicAttention,
    #[param]
    pub norm1: LayerNorm,
    #[param]
    pub norm2: LayerNorm,
    #[param]
    pub mlp: NomicMLP,
}

#[derive(Debug, ModuleParameters)]
pub struct NomicEncoder {
    #[param]
    pub layers: Vec<NomicBlock>,
}

#[derive(Debug, ModuleParameters)]
pub struct NomicBertModel {
    #[param]
    pub embeddings: NomicEmbeddings,
    #[param]
    pub emb_ln: LayerNorm,
    #[param]
    pub encoder: NomicEncoder,
}

impl NomicBertModel {
    pub fn new(config: &NomicBertConfig) -> Result<Self, Exception> {
        let hidden = config.n_embd;
        let inner = config.n_inner.unwrap_or(hidden * 4);
        let head_dim = hidden / config.n_head;
        let rotary_dim = (head_dim as f32 * config.rotary_emb_fraction) as i32;

        let layers = (0..config.n_layer)
            .map(|_| NomicBlock::new(config, hidden, inner, head_dim, rotary_dim))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            embeddings: NomicEmbeddings {
                word_embeddings: Embedding::new(config.vocab_size, hidden)?,
            },
            emb_ln: nn::LayerNormBuilder::new(hidden)
                .eps(config.layer_norm_epsilon)
                .build()?,
            encoder: NomicEncoder { layers },
        })
    }

    pub fn forward(
        &mut self,
        input_ids: &Array,
        attention_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let mut x = self.embeddings.word_embeddings.forward(input_ids)?;
        x = self.emb_ln.forward(&x)?;

        let mask = if let Some(am) = attention_mask {
            let am_f = am.as_type::<f32>()?;
            let ones = Array::from_f32(1.0);
            let neg = Array::from_f32(-10000.0);
            let m = ones.subtract(&am_f)?.multiply(&neg)?;
            let shape = am.shape();
            Some(m.reshape(&[shape[0], 1, 1, shape[1]])?)
        } else {
            None
        };

        for layer in &mut self.encoder.layers {
            x = layer.forward(&x, mask.as_ref())?;
        }

        Ok(x)
    }
}

impl NomicBlock {
    fn new(
        config: &NomicBertConfig,
        hidden: i32,
        inner: i32,
        head_dim: i32,
        rotary_dim: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            attn: NomicAttention::new(config, hidden, head_dim, rotary_dim)?,
            norm1: nn::LayerNormBuilder::new(hidden)
                .eps(config.layer_norm_epsilon)
                .build()?,
            norm2: nn::LayerNormBuilder::new(hidden)
                .eps(config.layer_norm_epsilon)
                .build()?,
            mlp: NomicMLP::new(hidden, inner, config)?,
        })
    }

    fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let attn_out = self.attn.forward(x, mask)?;
        let x = self.norm1.forward(&x.add(&attn_out)?)?;
        let mlp_out = self.mlp.forward(&x)?;
        self.norm2.forward(&x.add(&mlp_out)?)
    }
}

impl NomicAttention {
    fn new(
        config: &NomicBertConfig,
        hidden: i32,
        head_dim: i32,
        rotary_dim: i32,
    ) -> Result<Self, Exception> {
        Ok(Self {
            Wqkv: nn::LinearBuilder::new(hidden, 3 * hidden)
                .bias(config.qkv_proj_bias)
                .build()?,
            out_proj: nn::LinearBuilder::new(hidden, hidden)
                .bias(false)
                .build()?,
            rope: nn::RopeBuilder::new(rotary_dim)
                .base(config.rotary_emb_base)
                .traditional(config.rotary_emb_interleaved)
                .build()?,
            num_heads: config.n_head,
            head_dim,
        })
    }

    fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array, Exception> {
        let shape = x.shape();
        let (b, l) = (shape[0], shape[1]);
        let hidden = self.num_heads * self.head_dim;

        let qkv = self.Wqkv.forward(x)?;
        let parts = qkv.split(3, -1)?;
        let q = parts[0]
            .reshape(&[b, l, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let k = parts[1]
            .reshape(&[b, l, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = parts[2]
            .reshape(&[b, l, self.num_heads, self.head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;

        let q = self.rope.forward(RopeInput::from((&q,)))?;
        let k = self.rope.forward(RopeInput::from((&k,)))?;

        use mlx_rs::fast::ScaledDotProductAttentionMask;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let sdpa_mask: Option<ScaledDotProductAttentionMask> =
            mask.map(ScaledDotProductAttentionMask::Array);
        let attn = mlx_rs::fast::scaled_dot_product_attention(
            &q, &k, &v, scale, sdpa_mask, None::<&Array>,
        )?;

        let attn = attn
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, l, hidden])?;

        self.out_proj.forward(&attn)
    }
}

impl NomicMLP {
    fn new(hidden: i32, inner: i32, config: &NomicBertConfig) -> Result<Self, Exception> {
        Ok(Self {
            fc11: nn::LinearBuilder::new(hidden, inner)
                .bias(config.mlp_fc1_bias)
                .build()?,
            fc12: nn::LinearBuilder::new(hidden, inner)
                .bias(config.mlp_fc1_bias)
                .build()?,
            fc2: nn::LinearBuilder::new(inner, hidden)
                .bias(config.mlp_fc2_bias)
                .build()?,
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array, Exception> {
        let up = self.fc11.forward(x)?;
        let gate = nn::silu(self.fc12.forward(x)?)?;
        self.fc2.forward(&gate.multiply(&up)?)
    }
}

/// Mean-pool hidden states using attention mask, then L2-normalize.
pub fn mean_pool_normalize(
    hidden: &Array,
    attention_mask: &Array,
) -> Result<Array, Exception> {
    let shape = attention_mask.shape();
    let mask = attention_mask
        .as_type::<f32>()?
        .reshape(&[shape[0], shape[1], 1])?;

    let numerator = hidden.multiply(&mask)?.sum_axis(1, None)?;
    let denominator = mask.sum_axis(1, None)?;
    let eps = Array::from_f32(1e-9);
    let denominator = ops::maximum(&denominator, &eps)?;
    let pooled = numerator.divide(&denominator)?;

    let norm_sq = pooled.multiply(&pooled)?.sum_axis(-1, true)?;
    let norm = norm_sq.sqrt()?;
    let norm_eps = Array::from_f32(1e-12);
    let norm = ops::maximum(&norm, &norm_eps)?;
    pooled.divide(&norm)
}

// ─── Embedder wrapper ──────────────────────────────────────────────────

pub struct Embedder {
    model: NomicBertModel,
    tokenizer: Tokenizer,
}

impl Embedder {
    /// Load model from safetensors directory and tokenizer path.
    pub fn load(model_dir: &Path, tokenizer_path: &Path) -> Result<Self, String> {
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .map_err(|e| format!("Failed to read config.json: {}", e))?;
        let config: NomicBertConfig = serde_json::from_str(&config_str)
            .map_err(|e| format!("Failed to parse config.json: {}", e))?;

        let mut model = NomicBertModel::new(&config)
            .map_err(|e| format!("Failed to create model: {}", e))?;
        model
            .load_safetensors(model_dir.join("model.safetensors"))
            .map_err(|e| format!("Failed to load weights: {}", e))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {}", e))?;

        Ok(Self { model, tokenizer })
    }

    /// Embed a batch of texts. Returns one Vec<f32> (768-dim) per input.
    pub fn embed(&mut self, texts: &[String], is_query: bool) -> Result<Vec<Vec<f32>>, String> {
        let prefixed: Vec<String> = if is_query {
            texts.iter().map(|t| format!("{}{}", QUERY_PREFIX, t)).collect()
        } else {
            texts.to_vec()
        };

        let mut results = Vec::new();
        let batch_size = 32;

        for chunk in prefixed.chunks(batch_size) {
            let chunk_vec: Vec<String> = chunk.to_vec();
            let (input_ids, attention_mask) = self.tokenize_batch(&chunk_vec, MAX_LENGTH);

            let hidden = self
                .model
                .forward(&input_ids, Some(&attention_mask))
                .map_err(|e| format!("Forward pass failed: {}", e))?;
            let result = mean_pool_normalize(&hidden, &attention_mask)
                .map_err(|e| format!("Pooling failed: {}", e))?;
            result
                .eval()
                .map_err(|e| format!("Eval failed: {}", e))?;

            let data = result.as_slice::<f32>();
            let dims = result.shape();
            let n = dims[0] as usize;
            let d = dims[1] as usize;
            for i in 0..n {
                results.push(data[i * d..(i + 1) * d].to_vec());
            }
        }

        Ok(results)
    }

    fn tokenize_batch(&self, texts: &[String], max_len: usize) -> (Array, Array) {
        let encodings: Vec<_> = texts
            .iter()
            .map(|t| self.tokenizer.encode(t.as_str(), true).unwrap())
            .collect();

        let batch_size = encodings.len();
        let mut input_ids = vec![0i32; batch_size * max_len];
        let mut attention_mask = vec![0i32; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let len = ids.len().min(max_len);
            for j in 0..len {
                input_ids[i * max_len + j] = ids[j] as i32;
                attention_mask[i * max_len + j] = 1;
            }
        }

        let ids = Array::from_slice(&input_ids, &[batch_size as i32, max_len as i32]);
        let mask = Array::from_slice(&attention_mask, &[batch_size as i32, max_len as i32]);
        (ids, mask)
    }
}
