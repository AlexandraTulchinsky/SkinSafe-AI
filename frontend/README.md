## SkinSafe AI — Frontend (Next.js)

Minimal Next.js 15 app for scanning a product image and viewing `safe` vs `avoid` ingredients.

### Quick start
```bash
cd frontend
npm install
npm run dev
# open http://localhost:3000
```

Build & run:
```bash
npm run build
npm start
```

### Backend API
- Expects backend at `http://localhost:8000`.
- If your backend runs elsewhere, update the fetch URL in `app/page.tsx` (`/analyze`).

### Notes
- Tech: Next.js, React, Tailwind, Radix UI.
 - Use Node 18+ or 20+.

