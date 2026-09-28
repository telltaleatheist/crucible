export const UNCLAIMED_ENGINE = {
  role: 'engine',
  managed_by: null,
  pages_engine: {
    engine: null,
    installed: false,
    detail: 'dots-ocr has no cuda-linux block',
    request: {
      model: 'dots-ocr',
      dpi: 200,
      max_pixels: 11289600,
      max_tokens: 8192,
      temperature: 0.0,
      prompt: 'Please output the layout information from the PDF image.',
      dialect: 'dots-json',
      concurrency: 12,
      truncated_finish_reason: 'length',
    },
  },
};
