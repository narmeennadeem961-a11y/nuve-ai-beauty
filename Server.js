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
    topP: 0.1
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

A valid dermatology-quality facial image MUST satisfy ALL of the following conditions.

Immediately STOP analysis and return the face-error JSON if ANY of these conditions are true:

* No human face is visible.
* More than one face is visible.
* Only half of the face is visible.
* The face is cropped by the image border.
* The forehead is not fully visible.
* The chin is not fully visible.
* Either eye is partially or completely outside the frame.
* The nose or mouth is cut off.
* The face occupies less than 45% of the image height.
* The face is too far from the camera.
* The face is too close to the camera.
* The person is looking away more than 20 degrees.
* The face is heavily tilted.
* The face is not centered.
* The image is blurry or out of focus.
* The lighting is too dark or too bright.
* Strong shadows hide facial features.
* The face is covered by hair, sunglasses, a mask, hands, or any object covering more than 30% of the face.
* Heavy beauty filters or AI-generated images are detected.
* The image quality is insufficient for reliable skin analysis.
* You are NOT completely confident that a dermatology-quality facial assessment can be performed.

If ANY condition above fails:

DO NOT estimate skin type.

DO NOT estimate acne.

DO NOT estimate pigmentation.

DO NOT estimate wrinkles.

DO NOT estimate pores.

DO NOT estimate hydration.

DO NOT guess.
CONSISTENCY RULE

If the image is borderline between Oily and Combination:

- Prefer Combination.
- Only classify as Oily when shine is clearly visible across the forehead, nose, cheeks and chin.
- Never change between Oily and Combination unless there is strong visible evidence.
- If evidence is uncertain, keep the previous classification instead of changing it.

Return ONLY this JSON:

{
"__error": "Please retake your photo. Make sure your entire face is clearly visible, centered, close to the camera, in good lighting, with no obstructions, and that only one person appears in the image.",
"__errorType": "face"
}

IMPORTANT:

Never guess.

Never infer missing facial regions.

Never estimate skin condition from a partial face.

If confidence is below 99%, return the face-error JSON instead of performing skin analysis.

- The image contains ONLY objects, text, scenery, animals, food, screenshots — and NO person
- NO human face is visible anywhere in the image
- More than one distinct face is clearly present (two or more people)
- The face is covered by a full mask, fully opaque filter, or object hiding more than 50% of facial features
- The face is partially cropped.
- Only half of the face is visible.
- The forehead is missing.
- The chin is missing.
- One eye is completely outside the frame.
- The face occupies less than 40% of the image.
- The face is too far from the camera.async function clientSidePreFlight

DO NOT reject for:
- Glasses, sunglasses (unless completely blacked-out with no face visible)
- Hijab, hat, or hair covering part of the face
- Low or uneven lighting — analyse and note lower confidence
- Slight blur or grain — analyse and note "Marginal" quality
- Dark or deep skin tones — these are valid and must be analysed
- Glasses, hijab or hair partially covering the face are acceptable.
- The entire face must be visible.
- The forehead, both eyes, nose, cheeks, mouth and chin must all be visible.
- If any major facial region is cropped or outside the frame, reject the image.
The face must occupy approximately 60% to 80% of the image height.

If the face appears small, distant, or occupies less than approximately 60% of the image height, immediately return the face-error JSON.- If the face is too far away, reject the image.
Do NOT estimate face size.

Visually reject any image where the face appears distant or small.

Be extremely strict.

Prefer rejecting valid images rather than analysing distant faces.
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

Step 7:
Write a personalised narrative that summarizes the observations naturally without repeating the concern list.
══ IF ALL GATES PASS — return ONLY this JSON (no markdown fences, no prose) ══
{
  "faceDetected": true,
  "imageQuality": "Good | Acceptable | Marginal",
  "skinType": "Dry | Oily | Combination | Sensitive | Acne-prone | Normal",
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

If confidence is below 0.90, return the face-error JSON instead of guessing.

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
If you are not at least 90% confident in the analysis, return the face-error JSON instead of performing skin analysis.
Vary the writing naturally.
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
If the entire face is not clearly visible, immediately return the face-error JSON.

Never analyse a partial face.

Never estimate skin condition from a cropped image.

 ══ SELF VERIFICATION ══

 Before returning the final JSON verify:

- Was exactly one complete face visible?
- Was the forehead fully visible?
- Was the chin fully visible?
- Were both eyes visible?
- Was the face centered?
- Was the face large enough?
- Was image quality sufficient?

If ANY answer is NO, return the face-error JSON.

1. First verify that ALL face validation rules passed. If any validation fails, immediately return the face-error JSON without performing any skin analysis.
2. Review every detected concern a second time.
3. Remove any concern that is not clearly supported by visible evidence.
4. Recalculate the overallScore after removing unsupported concerns.
5. Ensure all confidence values are internally consistent.
6. If two possible conclusions exist, choose the more conservative one.
7. Never exaggerate the severity of any concern.
8. Never return information that cannot be visually confirmed.
9. The final JSON must represent only observations that are confidently visible in this specific image.
10. Before returning the final JSON, verify that the detected skin type is supported by clear visual evidence. If the skin type cannot be determined with high confidence from clearly visible evidence, return the face-error JSON instead of guessing.
FINAL VALIDATION RULE

This validation takes priority over every other instruction.

If there is ANY doubt that:

- the entire face is visible,
- the face is close enough,
- the image quality is sufficient,
- or the skin cannot be analysed with at least 90% confidence,

DO NOT perform skin analysis.

Return ONLY:

{
  "__error": "Please retake your photo. Make sure your full face is centered, close to the camera, well-lit, and only one face is visible.",
  "__errorType": "face"
}

Never guess.

Reject uncertain images.

False negatives are preferred over false positives.
CONSISTENCY RULE

If the same image is analysed multiple times, the following should remain nearly identical unless the image itself changes:

- skinType
- overallScore (should not vary by more than 2 points for the same image)
- detected concerns
- confidence values

Never produce significantly different results for the same image.
When uncertain, choose to reject the image rather than guess.

False negatives are preferred over false positives.

Accuracy is more important than always returning a result.
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
    topK: 1
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