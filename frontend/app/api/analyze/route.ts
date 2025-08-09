import { type NextRequest, NextResponse } from "next/server"

// Dummy Python API function
async function callPythonAnalysisAPI(imageData: string) {
  console.log("🐍 PYTHON API: Starting analysis...")
  console.log("🐍 PYTHON API: Image data length:", imageData.length)

  // Simulate Python API processing time
  await new Promise((resolve) => setTimeout(resolve, 3000))

  // Simulate Python API response - returns ingredients directly
  console.log("🐍 PYTHON API: Analysis complete")

  // In a real implementation, this would be:
  // const response = await fetch('http://your-python-api:8000/analyze', {
  //   method: 'POST',
  //   headers: { 'Content-Type': 'application/json' },
  //   body: JSON.stringify({ image: imageData })
  // })
  // return await response.json()

  return {
    success: true,
    product: "Moisturizing Face Cream",
    ingredients: {
      safe: [
        "Water (Aqua)",
        "Glycerin",
        "Hyaluronic Acid",
        "Niacinamide",
        "Ceramides",
        "Panthenol (Pro-Vitamin B5)",
        "Allantoin",
        "Sodium PCA",
      ],
      avoid: [
        "Coconut Oil (feeds malassezia)",
        "Oleic Acid (may irritate sensitive skin)",
        "Isopropyl Myristate (comedogenic)",
        "Lauric Acid (feeds malassezia)",
      ],
    },
    confidence: 0.94,
    processing_time_ms: 2847,
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { image, timestamp } = body

    console.log("🔍 API: Received request at:", new Date(timestamp).toISOString())
    console.log("🔍 API: Image data present:", !!image)
    console.log("🔍 API: Image data type:", typeof image)

    if (image) {
      console.log("🔍 API: Image data length:", image.length)
      console.log("🔍 API: Image data preview:", image.substring(0, 50) + "...")
    }

    // Validate image data
    if (!image || typeof image !== "string") {
      console.log("❌ API: Invalid image data")
      return NextResponse.json(
        {
          success: false,
          error: "Invalid image data",
          message: "No image data provided or invalid format.",
        },
        { status: 400 },
      )
    }

    // Call Python API with the captured image
    console.log("🔍 API: Calling Python analysis API...")
    const pythonResponse = await callPythonAnalysisAPI(image)
    console.log("🔍 API: Python API response:", pythonResponse)

    // Use Python API response directly
    const analysisResult = {
      ...pythonResponse, // Use all data from Python API
      metadata: {
        image_size: image.length,
        timestamp: timestamp,
        api_version: "v1.0",
      },
    }

    console.log("✅ API: Returning analysis result")
    return NextResponse.json(analysisResult)
  } catch (error) {
    console.error("❌ API Error:", error)
    return NextResponse.json(
      {
        success: false,
        error: "Analysis failed",
        message: "Unable to process the image. Please try again.",
        details: error.message,
      },
      { status: 500 },
    )
  }
}
