# scripts/build_unified.py
"""
Build script that combines both repos for mobile deployment
"""

import os
import shutil
import subprocess

def combine_repos():
    """Combine necessary files from both repos"""
    
    # Create build directory
    os.makedirs('build/chimera_mobile', exist_ok=True)
    
    # Copy core files from CHIMERA_Cognitive_Architecture
    shutil.copytree(
        '../CHIMERA_Cognitive_Architecture/chimera',
        'build/chimera_mobile/chimera_core',
        dirs_exist_ok=True
    )
    
    # Copy collective files from chimera-collective-server
    shutil.copytree(
        '../chimera-collective-server/chimera/collective',
        'build/chimera_mobile/chimera_core/collective',
        dirs_exist_ok=True
    )
    
    # Create mobile package
    create_mobile_package()

def create_mobile_package():
    """Package for mobile deployment"""
    
    # Create setup.py for mobile
    setup_content = '''
from setuptools import setup, find_packages

setup(
    name="chimera-mobile",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "websockets",
        "aiohttp",
        # Add other dependencies
    ],
    python_requires=">=3.9",
)
'''
    
    with open('build/chimera_mobile/setup.py', 'w') as f:
        f.write(setup_content)
    
    # Build wheel for mobile
    subprocess.run([
        'python', 'setup.py', 'bdist_wheel'
    ], cwd='build/chimera_mobile')

if __name__ == '__main__':
    combine_repos()
    print("✅ Mobile package built successfully!")
