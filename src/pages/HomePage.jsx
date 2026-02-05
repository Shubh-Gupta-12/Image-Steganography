import React from "react";

export default function HomePage({
  coverImage,
  secretImage,
  alpha,
  setAlpha,
  status,
  loading,
  handleImageChange,
  handleProcess,
}) {
  return (
    <main className="container">
      <section className="hero">
        <h2>Hide Intelligence <span className="highlight">Inside Images</span></h2>
        <p>Advanced Image Steganography</p>
        <p className="subtitle">
          Stegora securely embeds confidential visuals within images using precision blending, ensuring privacy without visible distortion.
        </p>
      </section>

      <section className="controls">
        <div className="upload-grid">
          <label htmlFor="coverInput" className="upload-card">
            <div className="upload-label">
              <svg
                className="upload-icon"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
              <p>Cover Image</p>
              <span className="card-description">The image that will carry the hidden content.</span>
              {coverImage && <span className="filename">{coverImage.name}</span>}
            </div>
            <input
              id="coverInput"
              type="file"
              accept="image/*"
              onChange={handleImageChange("cover")}
              className="hidden-input"
            />
          </label>

          <label htmlFor="secretInput" className="upload-card secret">
            <div className="upload-label">
              <svg
                className="upload-icon"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                />
              </svg>
              <p>Secret Image</p>
              <span className="card-description">The image you want to conceal securely.</span>
              {secretImage && <span className="filename">{secretImage.name}</span>}
            </div>
            <input
              id="secretInput"
              type="file"
              accept="image/*"
              onChange={handleImageChange("secret")}
              className="hidden-input"
            />
          </label>
        </div>

        <div className="alpha-control">
          <label htmlFor="alphaSlider">Blend Strength (Alpha)</label>
          <p className="alpha-description">
            Controls how deeply the secret is embedded. Lower values increase invisibility.
          </p>
          <div className="slider-group">
            <input
              id="alphaSlider"
              type="range"
              min="0.01"
              max="0.5"
              step="0.01"
              value={alpha}
              onChange={(e) => setAlpha(parseFloat(e.target.value))}
              className="slider"
              title="Recommended range: 0.05 – 0.15 for optimal stealth."
            />
            <span className="alpha-value">{alpha.toFixed(2)}</span>
          </div>
          <p className="alpha-tooltip">Recommended range: 0.05 – 0.15 for optimal stealth.</p>
        </div>

        <button
          onClick={handleProcess}
          disabled={!coverImage || !secretImage || loading}
          className="process-btn"
        >
          {loading ? "Processing..." : "Conceal & Reveal"}
        </button>

        <p className="status">{status}</p>
      </section>
    </main>
  );
}
