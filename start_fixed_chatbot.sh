#!/bin/bash

echo "🇸🇪 STARTING FIXED SWEDISH CHATBOT"
echo "=================================="
echo ""
echo "🔧 FIXES APPLIED:"
echo "✅ Better Whisper Swedish recognition (small model)"
echo "✅ Bellman Swedish model (much better conversation)"
echo "✅ Improved response formatting (no weird artifacts)"
echo "✅ Better audio processing settings"
echo "✅ Shorter, more natural responses"
echo ""

# Activate virtual environment
source chatbot_env/bin/activate

# Set HF token
export HF_TOKEN='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'

echo "🚀 Starting improved chatbot..."
echo "📱 Will open at: http://localhost:7861"
echo ""
echo "💡 TIPS FOR BEST RESULTS:"
echo "• Speak clearly and not too fast"
echo "• Use shorter sentences (5-10 words)"
echo "• Try: 'Hej, vad heter du?'"
echo "• Try: 'Hur mår du idag?'"
echo "• Try: 'Berätta om Sverige'"
echo ""

# Run the fixed chatbot
python swedish_chatbot_fixed.py