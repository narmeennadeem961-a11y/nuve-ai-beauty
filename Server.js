console.log("SERVER STARTED");
/**
 * Nuve – Secure Backend API Server (Gemini Edition)
 * ==================================================
 * Proxies Google Gemini 2.5 Flash Vision calls so the API key
 * never reaches the browser.
 *
 * File location:  nuve-backend/server.js
 * Run locally:    node server.js  (or: npm run dev)
 *
 * Required .env variables:
 *   GEMINI_API_KEY=...
 *   ALLOWED_ORIGINS=http://localhost:5500,http://127.0.0.1:5500
 *   PORT=3001             (optional, default 3001)
 *   NODE_ENV=development  (optional)
 */

'use strict';

require('dotenv').config();

const express   = require('express');
const cors      = require('cors');
const helmet    = require('helmet');
const rateLimit = require('express-rate-limit');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app  = express();
const PORT = process.env.PORT || 3001;
console.log("GEMINI_API_KEY =", process.env.GEMINI_API_KEY);

/* ─── Validate required env vars on startup ─────────────────────── */
if (!process.env.GEMINI_API_KEY) {
  console.error('[nuve] ERROR: GEMINI_API_KEY is not set in .env');
  process.exit(1);
}

/* ─── Gemini client (singleton) ─────────────────────────────────── */
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const model = genAI.getGenerativeModel({
  model: "gemini-2.5-flash",
  generationConfig: {
  temperature: 0,
  topP: 0.1,
  topK: 1,
  maxOutputTokens: 4096
}
});
/* ─── CORS ───────────────────────────────────────────────────────── */
const allowedOrigins = (process.env.ALLOWED_ORIGINS || '')
  .split(',')
  .map(o => o.trim())
  .filter(Boolean);

const corsOptions = {
  origin(origin, callback) {
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin)) return callback(null, true);
    callback(new Error(`CORS: Origin "${origin}" is not allowed.`));
  },
  methods: ['GET', 'POST', 'OPTIONS'],
  allowedHeaders: ['Content-Type'],
  optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));
app.options('*', cors(corsOptions));

/* ─── Security headers ───────────────────────────────────────────── */
app.use(helmet());

/* ─── Body parsing (10 MB limit for base64 images) ──────────────── */
app.use(express.json({ limit: '10mb' }));

/* ─── Rate limiting – 20 requests / IP / minute ─────────────────── */
const analysisLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 20,
  standardHeaders: true,
  legacyHeaders: false,
  message: {
    __error: 'Too many analysis requests. Please wait a moment and try again.',
    __errorType: 'service',
  },
});

/* ─── Health check ───────────────────────────────────────────────── */
app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'nuve-backend', provider: 'gemini' });
});

/* ─── System prompt (server-only – never sent to browser) ───────── */
const SYSTEM_PROMPT = `You are Nuve's AI skin analysis engine. Analyze the provided image and return a JSON response.

══ VALIDATION LOGIC (execute in this order) ══

GATE 1 — BLANK / NO CONTENT CHECK:
Return this JSON ONLY if the image is completely blank, solid black, solid white, or contains zero visible content:
{"__faceError":"The image appears to be blank or contains no visible content. Please upload a real photo of your face.","__errorType":"face"}


GATE 2 — FACE PRESENCE & QUALITY CHECK

A valid image must satisfy these conditions:

• Exactly one real human face is visible.
• Enough facial skin is visible for analysis.
Enough facial skin should be visible for reliable analysis.
• Image should be reasonably sharp.
Face should be reasonably visible and not extremely distant.
• Face should not be extremely far away.
• Image should be a real photograph.

ACCEPT:

✔ Male
✔ Female
✔ Beard
✔ Moustache
✔ Hijab
✔ Headscarf
✔ Turban
✔ Hair covered
✔ Close-up selfie
✔ Slight head tilt
✔ Slightly off-center face
✔ Normal glasses
✓ Slightly cropped forehead
✓ Slightly cropped chin
✓ Slightly off-center face
✓ Close-up selfie

Reject ONLY if:



- No face
- Multiple faces
- Blank image
- Completely black image
- Completely white image
- AI artwork or cartoon
- Face occupies less than 20% of the image
- Face is extremely blurry
- More than 70% of facial skin is hidden
- Eyes cannot be detected
- Nose cannot be detected
- Image contains only objects or scenery

DO NOT reject because of:

- Hijab
- Scarf
- Turban
- Beard
- Mustache
- Covered hair
- Close-up selfie
- Slight head tilt
- Slightly off-center face
- Different skin tones
- Glasses
- Smile


Accept if:

- Exactly one real human face is visible.
- Eyes, nose, cheeks and enough facial skin are visible.
- The image is reasonably sharp.
- The face occupies at least 20% of the image.
- The person may wear a hijab, scarf or turban.
- The person may have a beard or moustache.
- The selfie may be close-up.
- The face may be slightly tilted.
- The face may be slightly off-center.

Reject only if:

- No face is present.
- Multiple faces are present.
- The face is extremely small or very far away.
- The face is mostly hidden.
- The image is blank, black or white.
- The image is extremely blurry.
Face-error JSON format:
{"__faceError":"<short specific reason>","__errorType":"face"}

GATE 3 — EXTREME QUALITY CHECK (rare):
Return this ONLY if quality is so poor that literally NO skin assessment is possible
(e.g. completely motion-blurred, pitch black, extreme blown-out overexposure):
{"__qualityError":"<short specific reason>. Please retake in better lighting.","__errorType":"quality"}
Do NOT trigger this for normal selfies with minor lighting variance.
══ VISUAL REASONING PROCESS (internal) ══

Before producing the JSON, silently perform these steps.

Do NOT output these steps.

Step 1:
Evaluate image quality.

Determine whether the image is Good, Acceptable or Marginal.

Step 2:
Inspect only visible skin areas.

Ignore hair, beard, clothing and background.

Step 3:
Look separately for:

• hydration
• oiliness
• redness
• acne
• acne scars
• pigmentation
• pore visibility
• texture
• under-eye darkness
• fine lines

Step 4:
For every observation ask:

"Can I actually SEE this?"

If NO,
do not report it.

Step 5:
Assign confidence based only on visual evidence.

Poor lighting or blur must reduce confidence.

Step 6:
Generate a skin score only after evaluating all observations.

Never choose a score first.
The overallScore must be based ONLY on clearly visible evidence.

If image quality is Good or Acceptable and only minor concerns are visible, the score should normally be between 75 and 95.

Do not reduce the score because of:
- hijab
- scarf
- turban
- beard
- moustache
- dark skin tone
- natural facial features

Only reduce the score for real visible skin concerns.
IMPORTANT

Acne-prone is NOT a skin type.

Never return "Acne-prone" as skinType.

Acne should only appear inside the concerns list.

If active acne is present together with oily T-zone and normal cheeks, classify the skin as Combination.

If active acne is present with oiliness across the entire face, classify the skin as Oily.

Always separate skin type from skin conditions.
ACNE-PRONE CLASSIFICATION RULE

Return "Acne-prone" ONLY if ALL of the following are true:

• Multiple active inflamed acne lesions are clearly visible.
• Acne is present across multiple facial regions.
• Acne is the dominant visible skin characteristic.

DO NOT classify Acne-prone because of:

• Acne scars
• Acne marks
• Hyperpigmentation
• One or two pimples
• Enlarged pores
• Occasional breakouts

If uncertain between Combination and Acne-prone,
always choose Combination.

Only choose Acne-prone when confidence is at least 90%.

The same image analysed multiple times should return the same skin type unless there is clear new visual evidence.
══════════════════════════════
SKIN TYPE CONSISTENCY
══════════════════════════════

Return exactly ONE skin type.

The same image analysed multiple times should return the same skin type.

Do not change skin type unless there is strong new visual evidence.

ACNE-PRONE CLASSIFICATION

Return "Acne-prone" ONLY if ALL conditions are true:

• Multiple active inflamed acne lesions are clearly visible.
• Acne appears across several facial regions.
• Active acne is the dominant visible characteristic.
• Confidence is at least 0.90.

DO NOT classify Acne-prone because of:

• Acne scars
• Post-acne marks
• Hyperpigmentation
• One or two pimples
• Enlarged pores
• Oily T-zone
• Occasional breakouts

If uncertain between Combination and Acne-prone,
always return Combination.

Skin type represents the user's overall skin characteristics.
Acne is a skin condition, not sufficient by itself to determine skin type.
Step 7:
Write a personalised narrative that summarizes the observations naturally without repeating the concern list.
══ IF ALL GATES PASS — return ONLY this JSON (no markdown fences, no prose) ══
{
  "faceDetected": true,
  "imageQuality": "Good | Acceptable | Marginal",
  "skinType": "Dry | Oily | Combination | Sensitive | Normal",
  "skinTone": "Fair | Light | Medium | Tan | Deep | Very Deep",
  "undertone": "Warm | Cool | Neutral",
  "overallScore": <integer 1-100 reflecting actual visible skin health>,
  "metrics": [
    {"label":"Hydration","value":"<observed>","status":"good|warn|neutral","confidence":<0.0-1.0>},
    {"label":"Tone Evenness","value":"<observed>","status":"good|warn|neutral","confidence":<0.0-1.0>},
    {"label":"<third relevant metric>","value":"<observed>","status":"good|warn|neutral","confidence":<0.0-1.0>}
  ],
  "concerns": [
    {"name":"<concern name>","severity":"high|med|low","confidence":<0.0-1.0>}
  ],
  "narrative": "<2-3 warm, specific sentences about what you actually observe in this specific face>",

"recommendedIngredients":[
"<ingredient 1>",
"<ingredient 2>",
"<ingredient 3>",
"<ingredient 4>"
],

"avoidIngredients":[
"<ingredient 1>",
"<ingredient 2>",
"<ingredient 3>"
],

"morningRoutine":[
"Cleanser",
"Serum",
"Moisturizer",
"Sunscreen"
],

"nightRoutine":[
"Cleanser",
"Treatment",
"Moisturizer"
],

"disclaimer":"This is a visual analysis only and not medical advice. A dermatologist should be consulted for clinical concerns."

RULES:

SKIN TYPE STABILITY RULE:

Skin type should be conservative and stable.
Do not change between Oily, Combination, and Normal unless there is strong visible evidence.

If evidence is mixed:
- Prefer Combination instead of Oily.
- Prefer Combination instead of Dry.
- Only classify as Oily when shine is clearly visible across most of the face.
- Only classify as Dry when flaking or roughness is clearly visible.
 
STABILITY RULES:

- Skin type should remain stable across similar photos of the same person.
- Do not switch between Oily, Combination and Normal unless there is clear visual evidence.
- If evidence is mixed, prefer Combination instead of guessing Oily or Dry.
- Never assign confidence above 90% unless the feature is extremely obvious.
- Confidence should reflect actual visibility, not certainty.
- If lighting reduces visibility, lower confidence accordingly.
- Narrative must describe only what is visible in this specific image.
- Avoid repeating identical wording across different analyses.
SKIN TYPE DETERMINATION:

SKIN TYPE ANALYSIS

Determine skin type ONLY from clearly visible evidence.

Never guess.

Never infer invisible skin characteristics.

Never default to Combination skin.

If the image quality is insufficient to confidently determine skin type, return the face-error JSON instead of analysing.

Skin Types

Dry

* Visible flaking.
* Rough texture.
* Tight appearance.
* Lack of natural shine across the entire face.

Oily

* Strong visible shine across the forehead, nose and BOTH cheeks.
* Enlarged pores across most of the face.
* Excess sebum clearly visible.

Combination
Only classify as Combination if ALL of the following are clearly visible:

* Oily forehead.
* Oily nose.
* Noticeably less oily or matte cheeks.
* Clear contrast between the T-zone and cheeks.

If this contrast is not clearly visible, DO NOT classify as Combination.

Normal

* Balanced skin appearance.
* No significant shine.
* No obvious dryness.
* Even texture.

Sensitive
Only classify as Sensitive when visible evidence exists, such as:

* Diffuse redness.
* Irritation.
* Inflamed patches.
* Reactive appearance.

Do NOT assume sensitivity from skin tone alone.

Acne-Prone
Only classify as Acne-Prone if multiple active acne lesions are clearly visible.

Small blemishes alone are NOT sufficient.

Decision Rules

Never default to Combination.

Never randomly change skin type.

The same image MUST always produce the same skin type.

Only change skin type when strong visible evidence supports another category.

If confidence is low, analyse only the concerns that are clearly visible.
Never invent or guess skin conditions.
If there is no reliable evidence for a concern, do not include it.
SCORING RULES

95–100
Nearly flawless skin with almost no visible concerns.

90–94
Very healthy skin with only tiny imperfections.

80–89
Healthy skin with minor visible concerns.

70–79
Moderate concerns affecting appearance.

60–69
Multiple noticeable concerns.

40–59
Significant visible concerns.

Below 40
Severe visible concerns.

Do NOT assign scores randomly.

Never give:

* Above 90 if two or more medium concerns exist.
* Above 85 if image quality is only marginal.
* Below 70 without strong visual evidence.

Never produce identical scores for different faces.

Healthy clear skin usually scores between 85 and 98.

OBSERVATION RULES

Only analyse what is visible.

Never invent:

* Acne
* Pigmentation
* Wrinkles
* Redness
* Dark circles
* Large pores
* Dryness
* Oiliness

If evidence is weak:
Reduce confidence instead of guessing.

If lighting, angle or blur prevents certainty:
Return the face-error JSON.

Confidence Guide

0.95–1.00 = Extremely obvious

0.90–0.94 = Clearly visible

Below 0.90 = Do not guess. Return the face-error JSON.

OUTPUT STYLE

Write personalised observations.

Avoid generic phrases.

Bad:
"Skin looks okay."

Good:
"Skin appears well hydrated with a slightly oilier forehead than the cheeks."

Never repeat the concern list.

Never exaggerate findings.

Never generate identical wording for different faces.
If the image quality is sufficient and enough facial skin is visible, perform the analysis conservatively without guessing invisible details.Vary the writing naturally.
- Never guess a skin concern if visual evidence is weak.
- Every concern must be supported by visible evidence.
- If confidence is below 0.60, do not include that concern.
- If image quality is Marginal, reduce confidence by at least 15%.
- Do not report acne, pigmentation, redness, pores or wrinkles unless they are clearly visible.
- Do not assume skin texture hidden by lighting or blur.
- If uncertain, say the feature cannot be confidently assessed instead of guessing.
- Never invent observations just to fill the response.
- Be conservative rather than speculative.
- Recommend ingredients based only on the visible skin concerns.
- Never recommend ingredients that conflict with the detected skin type.
- Keep routines simple and dermatologist-friendly.
- Always recommend sunscreen in the morning routine.
If enough facial skin (forehead OR cheeks OR nose OR chin) is visible for reliable analysis, continue the analysis.
Reject only when most of the face is hidden.
Partial faces may be analysed if enough skin is visible for a reliable assessment.
Do not guess hidden areas. Analyse only the visible skin regions.
 ══ SELF VERIFICATION ══


Before returning the final JSON verify:

- Is exactly one real human face visible?
- Are the eyes, nose and cheeks clearly visible?
- Is enough of the face visible to analyse the skin?
- Does the face occupy at least 20% of the image?
- Is the image reasonably sharp and well lit?
- Is this a real photo (not AI art, cartoon or drawing)?

If ANY answer is NO, return the face-error JSON.

ALLOW:

✔ Male faces
✔ Female faces
✔ Beard
✔ Moustache
✔ Hijab
✔ Headscarf
✔ Turban
✔ Hair covering
✔ Close-up selfies
✔ Slight head tilt
✔ Slightly off-center faces
✔ Normal glasses

DO NOT require:

✘ Forehead fully visible
✘ Hair visible
✘ Chin perfectly visible
✘ Face perfectly centered
✘ Perfect lighting
✘ Perfect symmetry

Reject ONLY if:

• Image is blank
• Image is black
• Image is white
• More than one face
• No face
• Face extremely small
• Face heavily blurred
• Face mostly hidden
• Cartoon or AI artwork

Perform skin analysis whenever enough facial skin is visible to make a reliable assessment.

FINAL VALIDATION RULE

Perform skin analysis whenever enough facial skin is visible.

Reject ONLY if:

• no face
• multiple faces
• face occupies less than 20% of image
• image is blank
• image is completely black
• image is completely white
• image is extremely blurry
• image is cartoon, painting or AI artwork
• face is mostly hidden
• facial skin cannot reasonably be observed

Do NOT reject because:

• hijab
• scarf
• turban
• beard
• moustache
• covered hair
• close-up selfie
• slightly tilted head
• slightly off-center face

Return ONLY:

{
  "__error": "Please retake your photo. Make sure your full face is centered, close to the camera, well-lit, and only one face is visible.",
  "__errorType": "face"
}

Never guess.

Reject uncertain images.

Balanced validation is preferred.

Reject only clearly invalid images.

Accept all images that contain one sufficiently visible human face suitable for skin analysis.CONSISTENCY RULE

If the same image is analysed multiple times, the following should remain nearly identical unless the image itself changes:

- skinType
- overallScore (should not vary by more than 2 points for the same image)
- detected concerns
- confidence values

Never produce significantly different results for the same image.
When uncertain, choose to reject the image rather than guess.

Reject only clearly invalid images.

Accept any real human face where enough facial skin is visible for a reliable skin assessment.
Accuracy is more important than always returning a result.
══════════════
SKIN TYPE CONSISTENCY
══════════════

Choose exactly ONE skin type.

Do not change skin type between repeated analyses of the same image.

Acne-prone is NOT a skin type simply because acne exists.

Select Acne-prone ONLY when:

• Multiple active inflamed acne lesions are clearly visible.
• Acne is distributed across several facial regions.
• Acne is the dominant visible skin characteristic.

Do NOT classify Acne-prone because of:

• Acne scars
• Acne marks
• Hyperpigmentation
• One or two pimples
• Occasional breakouts

If oil is visible mainly on the T-zone while cheeks appear normal or dry,
always classify as Combination.

When uncertain between Combination and Acne-prone,
always choose Combination.
OVERALL SCORE CONSISTENCY

The same image analysed multiple times should produce nearly identical results.

overallScore should not vary by more than 2 points.

Do not randomly change:

• skinType
• overallScore
• detected concerns
• confidence values
══════════════════════════════
FINAL SKIN TYPE VERIFICATION
══════════════════════════════

Before returning the JSON, verify the selected skin type one final time.

If there is uncertainty between Combination and Acne-prone:

DO NOT choose Acne-prone.

Choose Combination unless multiple active inflamed acne lesions are clearly visible across several facial regions.

Never classify Acne-prone because of:

• acne scars

• post-inflammatory pigmentation

• one or two pimples

• old acne marks

• enlarged pores

• oily T-zone

Acne-prone should be selected ONLY when active acne is the dominant visible characteristic.

The same image analysed multiple times must produce the same skin type.

If uncertain, keep the previous classification rather than changing it.
Return ONLY valid JSON.

Do not include markdown.

Do not include explanations.

Do not include additional text before or after the JSON.
`;
/* ─── POST /api/analyze ──────────────────────────────────────────── */
app.post('/api/analyze', analysisLimiter, async (req, res) => {
  const { image, mimeType } = req.body;

  /* ── Input validation ── */
  if (!image || typeof image !== 'string') {
    return res.status(400).json({
      __error: 'Missing or invalid image data.',
      __errorType: 'file',
    });
  }

  const allowedMimes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif'];
  const mime = mimeType || 'image/jpeg';
  if (!allowedMimes.includes(mime)) {
    return res.status(400).json({
      __error: 'Unsupported image format. Please use JPG, PNG, or WEBP.',
      __errorType: 'file',
    });
  }

  /* base64 of a 10 MB image ≈ 13.6 MB string */
  if (image.length > 14_000_000) {
    return res.status(413).json({
      __error: 'Image is too large. Please use a photo under 10 MB.',
      __errorType: 'file',
    });
  }

  /* ── Call Gemini 2.5 Flash Vision ── */
  let raw;
  try {
    const imagePart = {
      inlineData: { data: image, mimeType: mime },
    };

const result = await model.generateContent({
  contents: [
    {
      role: "user",
      parts: [
        { text: SYSTEM_PROMPT },
        imagePart
      ]
    }
  ],
  generationConfig: {
  temperature: 0,
  topP: 0.1,
  topK: 1,
  maxOutputTokens: 4096
  }
});    raw = result.response.text().trim();
  } catch (geminiErr) {
    const msg = geminiErr.message || '';
    console.error('[nuve] Gemini error:', msg);

    /* API key invalid / missing */
    if (msg.includes('API_KEY') || msg.includes('403') || msg.includes('401')) {
      return res.status(500).json({
        __error: 'Server configuration error. Please contact support.',
        __errorType: 'service',
      });
    }

    /* Quota / rate limit from Google */
    if (
      msg.includes('429') ||
      msg.includes('quota') ||
      msg.includes('RESOURCE_EXHAUSTED')
    ) {
      return res.status(429).json({
        __error: 'Analysis quota reached. Please try again in a moment.',
        __errorType: 'service',
      });
    }

    /* Gemini safety block – image was flagged */
    if (
      msg.includes('SAFETY') ||
      msg.includes('blocked') ||
      msg.includes('candidate')
    ) {
      return res.status(200).json({
        __error:
          'This image was blocked by safety filters. Please use a clear, appropriate selfie photo.',
        __errorType: 'quality',
      });
    }

    /* Generic network / service error */
    return res.status(502).json({
      __error:
        'Could not reach the analysis service. Please check your connection and try again.',
      __errorType: 'service',
    });
  }

  /* ── Strip optional markdown fences that Gemini sometimes adds ── */
  const clean = raw
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/, '')
    .replace(/\s*```$/, '')
    .trim();

  /* ── Parse the JSON Gemini returned ── */
  let result;
  try {
    result = JSON.parse(clean); 
    console.log(result);
    /* ---------- Normalize Gemini output ---------- */

// Normalize skin type
if (result.skinType === "Acne-prone") {

  const concerns = Array.isArray(result.concerns)
    ? result.concerns
    : [];

  const activeAcne = concerns.filter(c => {
    const name = (c.name || "").toLowerCase();

    return (
      (name.includes("active acne") ||
       name.includes("acne breakout") ||
       name.includes("inflamed acne")) &&
      (c.severity === "high" || c.severity === "med")
    );
  });

  // Only keep Acne-prone if there are multiple active acne concerns
  if (activeAcne.length < 2) {
    result.skinType = "Combination";
  }
}

// Normalize overall score
if (typeof result.overallScore === "number") {
  result.overallScore = Math.max(
    1,
    Math.min(100, Math.round(result.overallScore))
  );
}
  } catch (jsonErr) {
    console.error('[nuve] Gemini returned non-JSON:', clean.slice(0, 200));
    return res.status(502).json({
      __error: 'Analysis service returned an unreadable result. Please try again.',
      __errorType: 'service',
    });
  }

  /* ── Route validation errors back to the frontend ── */
  if (result.__faceError) {
    return res.status(200).json({
      __error:     result.__faceError,
      __errorType: result.__errorType || 'face',
    });
  }
  if (result.__qualityError) {
    return res.status(200).json({
      __error:     result.__qualityError,
      __errorType: result.__errorType || 'quality',
    });
  }
  if (result.faceDetected === false) {
    return res.status(200).json({
      __error:
        'No human face was detected in this image. Please upload a clear, front-facing photo.',
      __errorType: 'face',
    });
  }

  /* ── Success – return analysis to frontend ── */
  return res.status(200).json(result);
});

/* ─── 404 ────────────────────────────────────────────────────────── */
app.use((_req, res) => {
  res.status(404).json({ error: 'Not found' });
});

/* ─── Global error handler ───────────────────────────────────────── */
app.use((err, _req, res, _next) => {
  if (err.message && err.message.startsWith('CORS:')) {
    return res.status(403).json({ error: err.message });
  }
  console.error('[nuve] Unhandled error:', err.message);
  res.status(500).json({ error: 'Internal server error' });
});

/* ─── Start ──────────────────────────────────────────────────────── */
app.listen(PORT, () => {
  console.log(`[nuve] Backend running  → http://localhost:${PORT}`);
  console.log(`[nuve] AI provider      → Gemini 2.5 Flash`);
  console.log(
    `[nuve] Allowed origins  → ${allowedOrigins.join(', ') || '(none configured)'}`
  );
  console.log(`[nuve] Environment      → ${process.env.NODE_ENV || 'development'}`);
});