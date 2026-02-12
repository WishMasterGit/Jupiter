Run the full processing pipeline

Usage: jupiter run [OPTIONS] [FILE]

Arguments:
  [FILE]  Input SER file

Options:
      --config <CONFIG>
          Pipeline config file (TOML)
  -v, --verbose
          Enable verbose output
      --device <DEVICE>
          Compute device (auto, cpu, gpu, cuda) [default: auto] [possible values: auto, cpu, gpu, cuda]
      --select <SELECT>
          Percentage of best frames to keep (1-100) [default: 25]
      --method <METHOD>
          Stacking method [default: mean] [possible values: mean, median, sigma-clip, multi-point]
      --sigma <SIGMA>
          Sigma threshold for sigma-clip stacking [default: 2.5]
      --sharpen <SHARPEN>
          Comma-separated wavelet sharpening coefficients
      --denoise <DENOISE>
          Comma-separated wavelet denoise thresholds
      --ap-size <AP_SIZE>
          Alignment point size in pixels (multi-point mode) [default: 64]
      --search-radius <SEARCH_RADIUS>
          Search radius around each AP for local alignment (multi-point mode) [default: 16]
      --min-brightness <MIN_BRIGHTNESS>
          Minimum mean brightness to place an alignment point (multi-point mode) [default: 0.05]
      --deconv <DECONV>
          Deconvolution method (rl or wiener)
      --psf <PSF>
          PSF model (gaussian, kolmogorov, airy) [default: gaussian]
      --psf-sigma <PSF_SIGMA>
          Gaussian PSF sigma in pixels [default: 2.0]
      --seeing <SEEING>
          Kolmogorov seeing FWHM in pixels [default: 3.0]
      --airy-radius <AIRY_RADIUS>
          Airy first dark ring radius in pixels [default: 2.5]
      --rl-iterations <RL_ITERATIONS>
          Richardson-Lucy iteration count [default: 20]
      --noise-ratio <NOISE_RATIO>
          Wiener noise-to-signal ratio [default: 0.001]
      --no-sharpen
          Disable sharpening
      --auto-stretch
          Auto histogram stretch after processing
      --gamma <GAMMA>
          Gamma correction after processing
  -o, --output <OUTPUT>
          Output file path
      --save-config <SAVE_CONFIG>
          Save effective config as TOML and exit without processing
  -h, --help
          Print help
