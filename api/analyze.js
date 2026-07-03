const { GoogleGenerativeAI } = require("@google/generative-ai");

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { image, mimeType } = req.body;

    if (!image) {
      return res.status(400).json({ error: "No image provided" });
    }

    // 👇 YAHAN tumhara pura analysis prompt aur Gemini code aayega.