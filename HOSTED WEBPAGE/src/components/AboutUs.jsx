import React from "react";
import { motion } from "framer-motion";

const AboutUs = () => {
  return (
    <section className="bg-black text-white py-20 px-4 md:px-16">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="max-w-5xl mx-auto text-center"
      >
        <h2 className="text-4xl font-bold mb-6 text-balance">About Us</h2>
        <p className="text-lg text-gray-400 mb-6">
          We are a team of passionate developers and researchers committed to bridging communication gaps through inclusive technology.
        </p>
        <p className="text-lg text-gray-400">
          Our Real-Time Sign Language Translator leverages the power of AI, computer vision, and modern web design to foster inclusivity and break down barriers—one gesture at a time.
        </p>
      </motion.div>
    </section>
  );
};

export default AboutUs;
