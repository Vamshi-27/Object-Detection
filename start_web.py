"""
Quick Web Interface Launcher
"""

import subprocess
import sys
import webbrowser
import threading
import time
import os

def main():
    print("🌐 YOLO Web Interface Launcher")
    print("=" * 40)
    print()
    
    # Check if Flask is installed
    try:
        import flask
        import flask_socketio
        print("✅ Flask dependencies found")
    except ImportError:
        print("❌ Flask not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "flask", "flask-socketio"], check=True)
        print("✅ Flask installed successfully")
    
    print("🚀 Starting YOLO Web Interface...")
    print("📱 Server will start at: http://localhost:5000")
    print("🌐 Your browser will open automatically")
    print()
    print("Features:")
    print("  • Drag & drop file upload")
    print("  • Real-time webcam detection")
    print("  • Live statistics")
    print("  • Model switching")
    print("  • Confidence adjustment")
    print()
    
    try:
        # Start web server in background
        def start_server():
            subprocess.run([sys.executable, "web_interface.py"], check=True)
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
        
        # Wait a moment then open browser
        print("⏳ Waiting for server to start...")
        time.sleep(4)
        
        print("🌐 Opening browser...")
        webbrowser.open('http://localhost:5000')
        
        print()
        print("🎯 Web interface is now running!")
        print("📋 Instructions:")
        print("  • Upload images or videos using drag & drop")
        print("  • Click 'Start Webcam' for real-time detection")
        print("  • Adjust confidence and model in the settings")
        print("  • View detection results and statistics in real-time")
        print()
        print("Press Ctrl+C to stop the server")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        print("💡 Try running: python web_interface.py directly")

if __name__ == "__main__":
    main()
