"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Camera, CheckCircle, XCircle, Zap, Brain, Shield, Sparkles } from "lucide-react"

interface AnalysisResult {
  success: boolean
  product?: string
  ingredients?: {
    safe: string[]
    avoid: string[]
  }
  confidence?: number
  processing_time_ms?: number
  metadata?: {
    image_size: number
    timestamp: number
    api_version: string
  }
  error?: string
  message?: string
}

export default function SkinSafeAI() {
  const [isScanning, setIsScanning] = useState(false)
  const [scanComplete, setScanComplete] = useState(false)
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)
  const [showCamera, setShowCamera] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [scanCount, setScanCount] = useState(0)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Load scan count from localStorage on component mount
  useEffect(() => {
    const savedCount = localStorage.getItem("skinsafe-scan-count")
    if (savedCount) {
      setScanCount(Number.parseInt(savedCount, 10))
    }
  }, [])

  // Save scan count to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem("skinsafe-scan-count", scanCount.toString())
  }, [scanCount])

  const startCamera = async () => {
    console.log("🔍 DEBUG: startCamera function called")

    try {
      // Step 1: Check if we're in a secure context
      console.log("🔍 DEBUG: Checking secure context...")
      console.log("🔍 DEBUG: location.protocol =", window.location.protocol)
      console.log("🔍 DEBUG: isSecureContext =", window.isSecureContext)

      // Step 2: Check if navigator exists
      console.log("🔍 DEBUG: Checking navigator...")
      console.log("🔍 DEBUG: navigator exists =", !!navigator)
      console.log("🔍 DEBUG: navigator.mediaDevices exists =", !!navigator.mediaDevices)
      console.log("🔍 DEBUG: getUserMedia exists =", !!navigator.mediaDevices?.getUserMedia)

      // Step 3: Check if getUserMedia is supported
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.log("❌ DEBUG: getUserMedia not supported")
        alert("Camera not supported on this device/browser. Using file upload instead.")
        handleFileUpload()
        return
      }

      console.log("✅ DEBUG: getUserMedia is supported")

      // Step 4: Show camera UI first, then request permissions
      console.log("🔍 DEBUG: Setting showCamera to true first...")
      setShowCamera(true)

      // Wait for next render cycle to ensure video element is created
      await new Promise((resolve) => setTimeout(resolve, 100))

      // Step 5: Check video element after UI update
      console.log("🔍 DEBUG: Checking video element after UI update...")
      console.log("🔍 DEBUG: videoRef.current exists =", !!videoRef.current)

      if (!videoRef.current) {
        console.log("❌ DEBUG: videoRef.current is still null after UI update")
        // Wait a bit more and try again
        await new Promise((resolve) => setTimeout(resolve, 200))
        console.log("🔍 DEBUG: Checking video element again...")
        console.log("🔍 DEBUG: videoRef.current exists =", !!videoRef.current)
      }

      // Step 6: Request camera permissions
      console.log("🔍 DEBUG: Requesting camera access...")
      console.log("🔍 DEBUG: Video constraints:", {
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })

      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "environment",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      })

      console.log("✅ DEBUG: Camera access granted")
      console.log("🔍 DEBUG: MediaStream:", mediaStream)
      console.log("🔍 DEBUG: Video tracks:", mediaStream.getVideoTracks())

      // Step 7: Set up video element
      if (videoRef.current) {
        console.log("🔍 DEBUG: Setting video source...")
        videoRef.current.srcObject = mediaStream
        setStream(mediaStream)

        console.log("🔍 DEBUG: Video element setup complete")

        // Step 8: Handle video loading
        videoRef.current.onloadedmetadata = () => {
          console.log("✅ DEBUG: Video metadata loaded")
          console.log("🔍 DEBUG: Video dimensions:", {
            videoWidth: videoRef.current?.videoWidth,
            videoHeight: videoRef.current?.videoHeight,
          })
          videoRef.current
            ?.play()
            .then(() => {
              console.log("✅ DEBUG: Video playing successfully")
            })
            .catch((playError) => {
              console.error("❌ DEBUG: Video play error:", playError)
            })
        }

        videoRef.current.onerror = (error) => {
          console.error("❌ DEBUG: Video element error:", error)
        }
      } else {
        console.error("❌ DEBUG: videoRef.current is still null after camera access")
        // Clean up the stream since we can't use it
        mediaStream.getTracks().forEach((track) => track.stop())
        setShowCamera(false)
        alert("Video element not ready. Please try again.")
      }
    } catch (error) {
      console.error("❌ DEBUG: Camera error caught:", error)
      console.error("❌ DEBUG: Error name:", error.name)
      console.error("❌ DEBUG: Error message:", error.message)
      console.error("❌ DEBUG: Error stack:", error.stack)

      // Reset UI state on error
      setShowCamera(false)

      // More specific error handling
      if (error.name === "NotAllowedError") {
        console.log("❌ DEBUG: Permission denied")
        alert("Camera access denied. Please allow camera permissions and try again, or use file upload instead.")
      } else if (error.name === "NotFoundError") {
        console.log("❌ DEBUG: No camera found")
        alert("No camera found on this device. Using file upload instead.")
      } else if (error.name === "NotSupportedError") {
        console.log("❌ DEBUG: Camera not supported")
        alert("Camera not supported on this browser. Using file upload instead.")
      } else {
        console.log("❌ DEBUG: Unknown error")
        alert("Camera error: " + error.message + ". Using file upload instead.")
      }

      // Fallback to file upload
      handleFileUpload()
    }
  }

  const handleFileUpload = () => {
    console.log("🔍 DEBUG: handleFileUpload called")
    const input = document.createElement("input")
    input.type = "file"
    input.accept = "image/*"
    input.capture = "environment"

    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      console.log("🔍 DEBUG: File selected:", file?.name)
      if (file) {
        const reader = new FileReader()
        reader.onload = (e) => {
          const imageData = e.target?.result as string
          console.log("🔍 DEBUG: File loaded, starting analysis...")
          console.log("🔍 DEBUG: File image data length:", imageData?.length)
          handleScanProduct(imageData)
        }
        reader.readAsDataURL(file)
      }
    }

    console.log("🔍 DEBUG: Triggering file input click")
    input.click()
  }

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
    setShowCamera(false)
  }

  const captureImage = async () => {
    if (videoRef.current && canvasRef.current) {
      console.log("🔍 DEBUG: Capturing image...")
      const canvas = canvasRef.current
      const video = videoRef.current
      const context = canvas.getContext("2d")

      if (!context) {
        console.error("❌ DEBUG: Could not get canvas context")
        return
      }

      // Set canvas dimensions to match video
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      console.log("🔍 DEBUG: Canvas dimensions:", {
        width: canvas.width,
        height: canvas.height,
      })

      // Draw the video frame to canvas
      context.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Convert canvas to base64 image data (JPEG format, 90% quality)
      const imageData = canvas.toDataURL("image/jpeg", 0.9)
      console.log("🔍 DEBUG: Image captured successfully")
      console.log("🔍 DEBUG: Image data length:", imageData.length)
      console.log("🔍 DEBUG: Image data format:", imageData.substring(0, 30) + "...")

      // Stop camera
      stopCamera()

      // Start analysis with captured image data
      await handleScanProduct(imageData)
    } else {
      console.error("❌ DEBUG: Video or canvas element not available")
    }
  }

  const handleScanProduct = async (imageData?: string) => {
    console.log("🔍 DEBUG: Starting analysis...")
    console.log("🔍 DEBUG: Image data provided:", !!imageData)
    console.log("🔍 DEBUG: Image data length:", imageData?.length || 0)

    setIsScanning(true)
    setScanComplete(false)

    try {
      const requestBody = {
        image: imageData || null,
        timestamp: Date.now(),
        source: imageData ? "camera_capture" : "fallback",
      }

      console.log("🔍 DEBUG: Sending request to API...")
      console.log("🔍 DEBUG: Request body keys:", Object.keys(requestBody))

      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      })

      console.log("🔍 DEBUG: API response status:", response.status)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      console.log("🔍 DEBUG: API response received:", result)
      console.log("🔍 DEBUG: Python API response:", result.python_api_response)
      console.log("🔍 DEBUG raw backend response:", result);   
      setAnalysisResult(result)
      setScanComplete(true)

      // Increment scan count on successful analysis
      if (result.success) {
        setScanCount((prev) => prev + 1)
        console.log("✅ DEBUG: Scan count incremented to:", scanCount + 1)
      }
    } catch (error) {
      console.error("❌ DEBUG: Analysis failed:", error)
      setAnalysisResult({
        success: false,
        error: "Analysis failed",
        message: "Unable to analyze the image. Please try again.",
      })
    } finally {
      setIsScanning(false)
    }
  }

  // Static preview data (shown before scanning)
  const previewData = {
    safe: ["Glycerin", "Hyaluronic Acid", "Niacinamide"],
    avoid: ["Coconut Oil (feeds malassezia)"],
  }

  // Determine if product is safe based on avoid ingredients
  const getProductSafety = () => {
    if (analysisResult?.success && analysisResult.ingredients) {
      const hasAvoidIngredients = (analysisResult.ingredients.avoid?.length ?? 0) > 0
      return {
        isSafe: !hasAvoidIngredients,
        text: hasAvoidIngredients ? "⚠️ Not Safe to Use" : "✓ Safe to Use",
        className: hasAvoidIngredients ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600",
      }
    }
    // Default for preview
    const hasAvoidIngredients = previewData.avoid.length > 0
    return {
      isSafe: !hasAvoidIngredients,
      text: hasAvoidIngredients ? "⚠️ Not Safe to Use" : "✓ Safe to Use",
      className: hasAvoidIngredients ? "bg-red-500 hover:bg-red-600" : "bg-green-500 hover:bg-green-600",
    }
  }

  const productSafety = getProductSafety()

  // Format scan count for display
  const formatScanCount = (count: number) => {
    if (count >= 1000000) {
      return `${(count / 1000000).toFixed(1)}M`
    } else if (count >= 1000) {
      return `${(count / 1000).toFixed(1)}K`
    }
    return count.toString()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-pink-50 to-blue-50">
      {/* Header */}
      <header className="p-4 text-center">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Sparkles className="w-6 h-6 text-purple-600" />
          <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            SkinSafe AI
          </h1>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        <div className="grid lg:grid-cols-2 gap-8 items-start">
          {/* Left Column */}
          <div className="space-y-8">
            {/* Badge */}
            <div className="animate-fade-in">
              <Badge variant="secondary" className="bg-orange-100 text-orange-800 border-orange-200">
                ⭐ AI-Powered Skin Analysis
              </Badge>
            </div>

            {/* Hero Text */}
            <div className="space-y-4 animate-slide-up">
              <h2 className="text-4xl lg:text-5xl font-bold leading-tight">
                FIGHT
                <br />
                MALASSEZIA
                <br />
                AND ECZEMA
                <br />
                <span className="bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 bg-clip-text text-transparent">
                  WITH SKINSAFE
                </span>
              </h2>
              <p className="text-gray-600 text-lg max-w-md">
                Instantly analyze product ingredients with AI to find skincare that's safe for sensitive skin
                conditions.
              </p>
            </div>

            {/* Scanner */}
            <Card className="animate-fade-in-up">
              <CardContent className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <Camera className="w-5 h-5 text-blue-600" />
                  <span className="font-semibold">PRODUCT SCANNER</span>
                </div>

                {scanComplete && analysisResult?.success && (
                  <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center gap-2 animate-fade-in">
                    <CheckCircle className="w-5 h-5 text-green-600" />
                    <span className="text-green-800 text-sm">Product scanned successfully! See results...</span>
                  </div>
                )}

                {showCamera ? (
                  <div className="space-y-6">
                    <div className="relative overflow-hidden rounded-2xl bg-black shadow-2xl">
                      <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-auto"
                        style={{ aspectRatio: "16/9" }}
                        onError={(e) => {
                          console.error("Video error:", e)
                          alert("Video playback error. Please try again.")
                          stopCamera()
                        }}
                      />

                      {/* Scanning frame overlay */}
                      <div className="absolute inset-6">
                        <div className="relative w-full h-full border-2 border-white/60 rounded-xl">
                          {/* Corner indicators */}
                          <div className="absolute top-0 left-0 w-6 h-6 border-l-4 border-t-4 border-purple-400 rounded-tl-lg"></div>
                          <div className="absolute top-0 right-0 w-6 h-6 border-r-4 border-t-4 border-purple-400 rounded-tr-lg"></div>
                          <div className="absolute bottom-0 left-0 w-6 h-6 border-l-4 border-b-4 border-purple-400 rounded-bl-lg"></div>
                          <div className="absolute bottom-0 right-0 w-6 h-6 border-r-4 border-b-4 border-purple-400 rounded-br-lg"></div>

                          {/* Instruction text */}
                          <div className="absolute top-4 left-4 bg-gradient-to-r from-purple-600 to-pink-600 text-white text-sm font-medium px-4 py-2 rounded-full shadow-lg backdrop-blur-sm">
                            📋 Point at ingredients list
                          </div>

                          {/* Center crosshair */}
                          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
                            <div className="w-8 h-8 border-2 border-purple-400 rounded-full bg-purple-400/20 animate-pulse"></div>
                          </div>
                        </div>
                      </div>

                      {/* Scanning animation line */}
                      <div className="absolute inset-0 pointer-events-none">
                        <div className="w-full h-0.5 bg-gradient-to-r from-transparent via-purple-400 to-transparent animate-scan"></div>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                      <Button
                        onClick={stopCamera}
                        variant="outline"
                        className="py-4 text-base font-medium border-2 hover:bg-gray-50 transition-all duration-200"
                      >
                        Cancel
                      </Button>
                      <Button
                        onClick={captureImage}
                        className="py-4 text-base font-medium bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105"
                      >
                        <Camera className="w-5 h-5 mr-2" />
                        Capture & Analyze
                      </Button>
                    </div>
                  </div>
                ) : (
                  <Button
                    onClick={() => {
                      console.log("🔍 DEBUG: Scan button clicked")
                      startCamera()
                    }}
                    disabled={isScanning}
                    className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white py-4 text-lg font-medium shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-105"
                  >
                    {isScanning ? (
                      <div className="flex items-center gap-3">
                        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        Analyzing...
                      </div>
                    ) : (
                      <div className="flex items-center gap-3">
                        <Camera className="w-6 h-6" />
                        Scan Product
                      </div>
                    )}
                  </Button>
                )}

                <canvas ref={canvasRef} className="hidden" />

                <p className="text-sm text-gray-500 text-center mt-4">
                  Point your camera at the ingredients list for instant analysis
                </p>
              </CardContent>
            </Card>

            {/* Features */}
            <div className="grid grid-cols-3 gap-4 animate-fade-in-up">
              <div className="text-center">
                <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  <Zap className="w-6 h-6 text-orange-600" />
                </div>
                <p className="text-sm font-medium">Instant Results</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-pink-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  <Brain className="w-6 h-6 text-pink-600" />
                </div>
                <p className="text-sm font-medium">AI-Powered</p>
              </div>
              <div className="text-center">
                <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-2">
                  <Shield className="w-6 h-6 text-blue-600" />
                </div>
                <p className="text-sm font-medium">Safe & Secure</p>
              </div>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-8">
            {/* Scan Results Preview (Static) or Actual Results */}
            <Card className="animate-slide-in-right">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold text-gray-800">
                    {analysisResult ? "Analysis Results" : "Scan Result Preview"}
                  </h3>
                  <Badge variant="default" className={productSafety.className}>
                    {productSafety.text}
                  </Badge>
                </div>

                {analysisResult?.success === false ? (
                  <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                      <XCircle className="w-4 h-4 text-red-600" />
                      <span className="font-medium text-red-800">Analysis Failed</span>
                    </div>
                    <p className="text-sm text-red-700">
                      {analysisResult.message || "Unable to analyze the image. Please try again."}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Safe Ingredients */}
                    <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <CheckCircle className="w-4 h-4 text-green-600" />
                        <span className="font-medium text-green-800">Safe Ingredients</span>
                        {analysisResult?.success && (
                          <span className="text-xs text-green-600">
                            ({analysisResult.ingredients?.safe?.length ?? 0} found)
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-green-700">
                      {analysisResult?.success && analysisResult.ingredients
                        ? (analysisResult.ingredients.safe?.length ?? 0) > 0
                          ? analysisResult.ingredients.safe.join(", ")
                          : "No safe ingredients found."
                        : previewData.safe.join(", ")}
                    </p>

                    </div>

                    {/* Avoid Ingredients */}
                    <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                      <div className="flex items-center gap-2 mb-2">
                        <XCircle className="w-4 h-4 text-red-600" />
                        <span className="font-medium text-red-800">Avoid</span>
                        {analysisResult?.success && (
                          <span className="text-xs text-red-600">
                            ({analysisResult.ingredients?.avoid?.length ?? 0} found)
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-red-700">
                      {analysisResult?.success && analysisResult.ingredients
                        ? (analysisResult.ingredients.avoid?.length ?? 0) > 0
                          ? analysisResult.ingredients.avoid.join(", ")
                          : "No avoid ingredients found."
                        : previewData.avoid.join(", ")}
                    </p>

                    </div>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* How It Works */}
            <Card className="animate-fade-in-up">
              <CardContent className="p-6">
                <h3 className="font-semibold text-gray-800 mb-4">How It Works</h3>
                <div className="space-y-4">
                  <div className="flex gap-3">
                    <div className="w-6 h-6 bg-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      1
                    </div>
                    <div>
                      <p className="font-medium">Scan Product</p>
                      <p className="text-sm text-gray-600">Point camera at ingredients list</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="w-6 h-6 bg-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      2
                    </div>
                    <div>
                      <p className="font-medium">AI Analysis</p>
                      <p className="text-sm text-gray-600">Smart AI agents check each ingredient</p>
                    </div>
                  </div>
                  <div className="flex gap-3">
                    <div className="w-6 h-6 bg-purple-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                      3
                    </div>
                    <div>
                      <p className="font-medium">Get Results</p>
                      <p className="text-sm text-gray-600">Instant safety report</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Stats */}
            <div className="grid grid-cols-2 gap-4 animate-fade-in-up">
              <Card>
                <CardContent className="p-6 text-center">
                  <div className="text-3xl font-bold text-purple-600 mb-1">{formatScanCount(scanCount)}+</div>
                  <div className="text-sm text-gray-600">Products Scanned</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-6 text-center">
                  <div className="text-3xl font-bold text-purple-600 mb-1">98%</div>
                  <div className="text-sm text-gray-600">Accuracy Rate</div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      <style jsx global>{`
        @keyframes fade-in {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        
        @keyframes slide-up {
          from { 
            opacity: 0;
            transform: translateY(20px);
          }
          to { 
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes fade-in-up {
          from { 
            opacity: 0;
            transform: translateY(30px);
          }
          to { 
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes slide-in-right {
          from { 
            opacity: 0;
            transform: translateX(30px);
          }
          to { 
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        @keyframes scan {
          0% { transform: translateY(0); }
          100% { transform: translateY(400px); }
        }
        
        .animate-fade-in {
          animation: fade-in 0.6s ease-out;
        }
        
        .animate-slide-up {
          animation: slide-up 0.8s ease-out;
        }
        
        .animate-fade-in-up {
          animation: fade-in-up 0.8s ease-out;
        }
        
        .animate-slide-in-right {
          animation: slide-in-right 0.8s ease-out;
        }
        
        .animate-scan {
          animation: scan 2s linear infinite;
        }
      `}</style>
    </div>
  )
}
