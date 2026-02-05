import React from "react";

export default function AboutPage() {
  return (
    <main className="container">
      <section className="about-hero">
        <h2><span style={{color: 'var(--text-main)', background: 'none', WebkitBackgroundClip: 'unset', WebkitTextFillColor: 'var(--text-main)'}}>Invisible</span> Intelligence</h2>
        <p>
          Stegora is an image steganography tool that hides one image inside another 
          using alpha blending combined with optional VAE-based denoising from 
          Stable Diffusion models.
        </p>
      </section>

      <section className="features-grid">
        <div className="feature-card">
          <div className="feature-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
          </div>
          <h3 className="feature-title">Image-in-Image Hiding</h3>
          <p className="feature-desc">
            Hide a secret image within a cover image using alpha blending. 
            The alpha value controls how visible the secret is — lower values make it harder to detect.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"></path>
            </svg>
          </div>
          <h3 className="feature-title">Encode & Decode</h3>
          <p className="feature-desc">
            Encode your secret into a stego image, then decode it back using the same 
            cover image and alpha value. Both images are required for extraction.
          </p>
        </div>

        <div className="feature-card">
          <div className="feature-icon">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
              <polyline points="2 17 12 22 22 17"></polyline>
              <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
          </div>
          <h3 className="feature-title">VAE Denoising (Optional)</h3>
          <p className="feature-desc">
            When a Stable Diffusion model is loaded, the VAE encoder-decoder helps 
            smooth the output. Falls back to Gaussian blur when the model isn't available.
          </p>
        </div>
      </section>
    </main>
  );
}
