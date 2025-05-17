import React from 'react';
import { motion } from 'framer-motion';

const CallToAction = () => {
  return (
    <section className="w-full px-6 py-16 flex justify-center items-center bg-black">
      <motion.div
        className="relative bg-gradient-to-br from-red-600/20 to-red-900/30 backdrop-blur-xl border border-red-500/30 rounded-3xl p-8 md:p-12 shadow-[0_0_40px_rgba(255,0,0,0.2)] text-center max-w-3xl w-full overflow-hidden"
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
      >
        {/* Glow ring animation */}
        <div className="absolute -inset-0.5 bg-gradient-to-br from-red-500 via-red-700 to-red-900 rounded-3xl blur-xl opacity-25 animate-pulse"></div>

        {/* Content layer */}
        <div className="relative z-10">
          <h3 className="text-3xl md:text-4xl font-bold text-white drop-shadow-lg mb-4">
            Unlock Real-Time Sign Language Translation
          </h3>
          <p className="text-base md:text-lg text-gray-300 mb-6">
            Step into the future of communication. Our AI translates ASL instantly — try it now and feel the magic.
          </p>
          <a
            href="/login"
            className="inline-block px-8 py-3 text-lg font-semibold text-white bg-red-600 rounded-full shadow-lg hover:shadow-red-500/50 hover:scale-105 transition-transform duration-300"
          >
            Try It Now
          </a>
        </div>
      </motion.div>
    </section>
  );
};

export default CallToAction;
