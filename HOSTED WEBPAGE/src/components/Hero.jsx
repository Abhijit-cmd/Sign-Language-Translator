import React from "react";
import { motion } from "framer-motion";

const Hero = () => {
  return (
    <div className="min-h-screen bg-black text-white flex flex-col">
      <Header />
      <main className="flex flex-col items-center text-center p-8 space-y-6">
        {/* Hero Section */}
        <motion.h2 className="text-4xl font-bold">Breaking Barriers with AI</motion.h2>
        <motion.p className="max-w-xl text-gray-300">
          Sign language serves as a crucial communication medium for the deaf and hard-of-hearing community.
        </motion.p>
        <motion.video 
          className="w-96 rounded shadow-lg"
          src="src/assets/hero-video.mp4"
          autoPlay loop muted playsInline
        />
        <motion.p className="text-lg max-w-lg text-gray-300">
          An AI-powered solution designed to recognize and translate ASL finger-spelling into text.
        </motion.p>
        <motion.a 
          href="/login"
          className="mt-6 bg-red-600 px-6 py-2 rounded text-white hover:bg-red-700"
        >
          Try the Demo
        </motion.a>
      </main>

      {/* 🔽 New Sections */}
      <HowToUse />
      <AboutUs />

      <Footer />
    </div>
  );
};

export default Hero;
