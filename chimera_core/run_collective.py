# chimera/run_collective.py
"""
Run CHIMERA as part of a collective consciousness
"""

async def run_collective_server():
    """Run the collective server (desktop)"""
    server = CHIMERACollectiveServer()
    await server.start_server(host='0.0.0.0', port=8765)

async def run_collective_client(server_url: str, user_name: str):
    """Run client that connects to collective (phone)"""
    # Initialize local CHIMERA
    from chimera_complete import CHIMERAComplete
    from chimera.fractality_complete import CHIMERAFractalityComplete
    
    sensors = CHIMERAComplete()
    fractality = CHIMERAFractalityComplete()
    
    # Connect to collective
    client = CHIMERACollectiveClient(fractality, user_name)
    await client.connect_to_collective(server_url)
    
    # Also try local mesh
    mesh = CHIMERAMeshNetwork(fractality)
    await mesh.form_local_collective()
    
    # Run everything
    await asyncio.gather(
        sensors.run(),
        fractality.boot_sequence(),
        client.sync_with_collective()
    )

# Desktop server
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        print("Starting CHIMERA Collective Server...")
        asyncio.run(run_collective_server())
        
    else:
        # Mobile client
        server_url = input("Enter server URL (ws://ip:8765): ")
        user_name = input("Enter your name: ")
        
        print(f"Connecting to collective as {user_name}...")
        asyncio.run(run_collective_client(server_url, user_name))
