import React from 'react';
import Header from '../components/Header';
import Footer from '../components/Footer';
import HowToUse from '../components/HowToUse';
import AboutUs from '../components/AboutUs';
import CallToAction from '../components/CTA';
import GlassCTA from '../components/GlassCTA';
import { motion } from 'framer-motion';

const Homepage = () => {
  return (
    <div
      className="min-h-screen bg-black text-white flex flex-col bg-cover bg-center bg-no-repeat"
      style={{ backgroundImage: "url('https://media-hosting.imagekit.io/8bfd284a18a24e75/ds.png?Expires=1839249320&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=hFebof4BsovkAe-i4atZ1Z4FATpFJLjnNEtm8MHYyRoEJC7Tx2WS7LTs66NS0gFf4AsqbVcbZvbO7-ue9dyhIlEQJjZvL2W~PGz4sSsaPyig1p0xFzQEID8wpmh5G2ChxutKvb1KY9PgO0A5DWp7sBCNbrtQSBcauF5V73o0EGsAktTJHnFHGSwZ9-13uSIUP2D1OP0cCICct2WYiNIL5pA3CYnim9RBJhsD0BYkB5d0OnTDQ8Q74KSPaQhE6TLeSG6cygffEbyIgm8ReHUQ~ksS0CNEVygb7jlldtoGc59A~xNcclstDmSDGg4Ts7CfbdniqZxCZq-WPdOCqSlctQ__')" }}
    >
      <Header />
      <main className="flex flex-col items-center text-center px-6 md:px-20 py-12 md:py-24 space-y-10 backdrop-blur-sm bg-black/70">
    <GlassCTA />
        {/* Hero Section */}
        <motion.h2 
          className="text-4xl sm:text-5xl md:text-6xl font-bold leading-tight md:leading-snug"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          Sign Language <span className="text-red-600">Translator</span>
        </motion.h2>

        <motion.p 
          className="text-base sm:text-lg md:text-xl max-w-3xl text-gray-300"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.8 }}
        >
          Sign language serves as a crucial communication medium for the deaf and hard-of-hearing community.
        </motion.p>

        <motion.video 
          className="w-full max-w-5xl aspect-video rounded-xl shadow-xl border border-gray-700"
          // src="https://media-hosting.imagekit.io/5e383f5a1cd44628/hero-video.mp4?Expires=1839249320&Key-Pair-Id=K2ZIVPTIP2VGHC&Signature=Hvw8x0uwN1bwLhf0AvsoZx32pST-NTokh98eeJsi72eVnF9PGvKj8ib9TPQUSks5k~tCrlxHXZRHu-SxBdRGnzca2ADaawur0-qSTBar-w7N8XLUi6bLVxBk5~BMCqgZnQkwYShCEJqseiDsQ0YrFjgjj3S6sfB0FhJtGu9T2pS~mvB8gb5QLp2gLLw9WS49fKH09fVijTkOVfmnLk12d4EiB1ICmuqZKWbZ2qLt4FYh5TmMLA5BUSQpwIoAQLi4bKeA~I0F7hf09m00qNj3cDMJ6lio13odSjwAHH6aLto3ntrj8xS8AzeVNioddoGMziUZU7XeHrccCY~gsWkuSg__"
          src="\hero-video.mp4"
          autoPlay loop muted playsInline
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 1 }}
        />

        <motion.p 
          className="text-base sm:text-lg md:text-xl max-w-2xl text-gray-300"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.8 }}
        >
          An AI-powered solution designed to recognize and translate ASL finger-spelling into text.
        </motion.p>

        <motion.a 
          href="/login"
          className="inline-block bg-red-600 hover:bg-red-700 transition px-8 py-3 rounded-full font-semibold shadow-lg"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Try the Demo
        </motion.a>
      </main>

      {/* Interactive Sections */}
      <HowToUse
        steps={[
          {
            title: "Step 1",
            description: "Enable your webcam and position your hand within the frame.",
            icon: "🎥",
          },
          {
            title: "Step 2",
            description: "Start signing in ASL using finger-spelling gestures.",
            icon: "🤟",
          },
          {
            title: "Step 3",
            description: "Watch the real-time translation appear on your screen.",
            icon: "💬",
          },
        ]}
      />

      <AboutUs />
      <CallToAction />
      <Footer />
    </div>
  );
};

export default Homepage;
