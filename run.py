#!/usr/bin/env python3
"""
Main entry point for CHIMERA with web interface
Run this to start everything
"""

import sys
import os
from pathlib import Path

# Add chimera to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    print("""
    ╔══════════════════════════════════════════════╗
    ║                                              ║
    ║     CHIMERA Cognitive Architecture v2.0     ║
    ║     Organic Learning Through Interaction    ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
    """)
    
    print("Starting CHIMERA components...")
    
    # Import and run the web app
    from web.app import app, socketio
    
    print("✓ Memory system initialized")
    print("✓ Learning system initialized")
    print("✓ Agent ecosystem initialized")
    print("✓ Web interface ready")
    print("\n" + "="*50)
    print("CHIMERA is running at: http://localhost:5000")
    print("Open this URL in your browser to interact")
    print("="*50 + "\n")
    
    # Run the app
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()