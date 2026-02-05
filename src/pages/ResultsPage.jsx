import React from "react";

export default function ResultsPage({ coverImage, secretImage, stegoImage, decodedImage }) {
  // Helper to create a download link content
  const downloadIcon = (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{marginRight: '8px'}}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="7 10 12 15 17 10"></polyline>
      <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
  );

  // Badge styles
  const purpleBadge = { background: 'rgba(139, 92, 246, 0.15)', color: '#8b5cf6' };
  const greyBadge = { background: 'var(--bg-input)', color: 'var(--text-secondary)' };
  const greenBadge = { background: 'rgba(34, 197, 94, 0.15)', color: '#22c55e' };

  return (
    <main className="container">
      <section className="results-header">
        <h2>Production <span style={{background: 'linear-gradient(135deg, #8b5cf6 0%, #a78bfa 50%, #c084fc 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text'}}>Gallery</span></h2>
        <p>Your processed outputs are ready for review and distribution.</p>
      </section>

      <div className="results-grid">
        {/* Cover Input */}
        <div className="result-card">
          <div className="result-header">
            <span className="result-title">Cover Image</span>
            <span className="result-badge" style={purpleBadge}>Source</span>
          </div>
          <div className="result-image-container">
            {coverImage ? (
              <img
                src={URL.createObjectURL(coverImage)}
                alt="Cover Source"
                className="result-image"
              />
            ) : (
              <div className="empty-state">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{opacity: 0.3}}>
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                  <circle cx="8.5" cy="8.5" r="1.5"></circle>
                  <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                <span>No source selected</span>
              </div>
            )}
          </div>
          <div className="action-row">
            {/* Disabled or info button could go here */}
          </div>
        </div>

        {/* Secret Input */}
        <div className="result-card">
          <div className="result-header">
            <span className="result-title">Secret Image</span>
            <span className="result-badge" style={purpleBadge}>Hidden</span>
          </div>
          <div className="result-image-container">
            {secretImage ? (
              <img
                src={URL.createObjectURL(secretImage)}
                alt="Secret Source"
                className="result-image"
              />
            ) : (
              <div className="empty-state">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{opacity: 0.3}}>
                  <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                  <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                </svg>
                <span>No secret selected</span>
              </div>
            )}
          </div>
        </div>

        {/* Stego Output */}
        <div className="result-card full">
          <div className="result-header">
            <span className="result-title">Steganographic Output</span>
            <span className="result-badge" style={stegoImage ? greenBadge : greyBadge}>Encoded</span>
          </div>
          <div className="result-image-container" style={{minHeight: '350px'}}>
            {stegoImage ? (
              <img src={stegoImage} alt="Stego Output" className="result-image" />
            ) : (
              <div className="empty-state">
                <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round" style={{opacity: 0.3}}>
                  <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                  <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                  <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                <span>Process the images to see the result</span>
              </div>
            )}
          </div>
          <div className="action-row">
            {stegoImage && (
              <a href={stegoImage} download="stego-result.png" className="Download-btn">
                {downloadIcon} Download Encoded Image
              </a>
            )}
          </div>
        </div>

        {/* Decoded Output */}
        <div className="result-card full">
          <div className="result-header">
            <span className="result-title">Decoded Verification</span>
            <span className="result-badge" style={decodedImage ? greenBadge : greyBadge}>Restored</span>
          </div>
          <div className="result-image-container" style={{minHeight: '350px'}}>
            {decodedImage ? (
              <img src={decodedImage} alt="Decoded Output" className="result-image" />
            ) : (
              <div className="empty-state">
                <span>Waiting for encoding to complete...</span>
              </div>
            )}
          </div>
          <div className="action-row">
            {decodedImage && (
              <a href={decodedImage} download="decoded-secret.png" className="Download-btn" style={{background: 'var(--bg-input)', color: 'var(--text-main)'}}>
                {downloadIcon} Download Restored Secret
              </a>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
