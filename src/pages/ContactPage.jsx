import React, { useState } from "react";

export default function ContactPage() {
  const [openFaq, setOpenFaq] = useState(null);

  const faqs = [
    {
      q: "What is image steganography?",
      a: "Image steganography is the practice of hiding secret information within an ordinary image. Stegora uses alpha blending to embed one image inside another, making the hidden content virtually invisible to the naked eye."
    },
    {
      q: "How does the alpha value work?",
      a: "The alpha value (0.05 - 0.15 recommended) controls how deeply the secret image is embedded. Lower values make the secret harder to detect but may reduce decoded quality. Higher values preserve more detail but may be more visible."
    },
    {
      q: "What image formats are supported?",
      a: "Stegora supports common image formats including JPEG, PNG, and WebP. For best results, use PNG format as it preserves quality without compression artifacts."
    },
    {
      q: "Do I need to save the cover image?",
      a: "Yes! To decode the hidden image later, you need both the original cover image and the same alpha value used during encoding. Keep these safe."
    },
    {
      q: "Is my data secure?",
      a: "All image processing happens on the server with no data stored permanently. Your images are processed in memory and immediately discarded after the response is sent."
    },
    {
      q: "Why is the decoded image blurry?",
      a: "Some quality loss is expected with steganography. Using higher alpha values (0.1-0.15) improves decoded quality. The secret image is also enhanced automatically to improve visibility."
    }
  ];

  const toggleFaq = (index) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  return (
    <main className="container">
      <section className="about-hero">
        <h2>
          <span style={{color: 'var(--text-main)', background: 'none', WebkitBackgroundClip: 'unset', WebkitTextFillColor: 'var(--text-main)'}}>
            Contact & Support
          </span>
        </h2>
        <p>
          Have questions about Stegora? Check out our frequently asked questions below 
          or reach out to the development team.
        </p>
      </section>

      <section className="contact-info" style={{
        background: 'var(--card-bg)',
        borderRadius: '16px',
        padding: '2rem',
        marginBottom: '2rem',
        textAlign: 'center'
      }}>
        <div style={{ marginBottom: '1rem' }}>
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
            <circle cx="9" cy="7" r="4"></circle>
            <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
            <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
          </svg>
        </div>
        <h3 style={{ color: 'var(--text-main)', marginBottom: '0.5rem' }}>Group Project</h3>
        <p style={{ color: 'var(--text-muted)' }}>
          Stegora is developed as an academic group project exploring image steganography 
          techniques using modern web technologies and machine learning.
        </p>
      </section>

      <section className="faq-section">
        <h3 style={{ 
          color: 'var(--text-main)', 
          marginBottom: '1.5rem',
          fontSize: '1.5rem',
          textAlign: 'center'
        }}>
          Frequently Asked Questions
        </h3>
        <div className="faq-list" style={{ maxWidth: '700px', margin: '0 auto' }}>
          {faqs.map((faq, index) => (
            <div 
              key={index} 
              className="faq-item" 
              style={{
                background: 'var(--card-bg)',
                borderRadius: '12px',
                marginBottom: '0.75rem',
                overflow: 'hidden',
                border: openFaq === index ? '1px solid var(--accent)' : '1px solid transparent',
                transition: 'border 0.2s ease'
              }}
            >
              <button
                onClick={() => toggleFaq(index)}
                style={{
                  width: '100%',
                  padding: '1rem 1.25rem',
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  textAlign: 'left',
                  color: 'var(--text-main)',
                  fontSize: '1rem',
                  fontWeight: '500'
                }}
              >
                {faq.q}
                <svg 
                  width="20" 
                  height="20" 
                  viewBox="0 0 24 24" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2"
                  style={{
                    transform: openFaq === index ? 'rotate(180deg)' : 'rotate(0)',
                    transition: 'transform 0.2s ease',
                    flexShrink: 0,
                    marginLeft: '1rem'
                  }}
                >
                  <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
              </button>
              {openFaq === index && (
                <div style={{
                  padding: '0 1.25rem 1rem',
                  color: 'var(--text-muted)',
                  lineHeight: '1.6'
                }}>
                  {faq.a}
                </div>
              )}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
