# chimera/visualization/learning_dashboard.py
"""
Real-time visualization of CHIMERA's learning
"""

class LearningDashboard:
    def __init__(self, council):
        self.council = council
        
    def display_learning_state(self):
        """Show what CHIMERA has learned"""
        
        print("\n📊 LEARNING STATE DASHBOARD")
        print("=" * 60)
        
        # Working Memory contents
        wm = self.council.modules['memory_wm']
        print(f"\n💾 WORKING MEMORY ({len(wm.memory_buffer)}/{wm.capacity})")
        for item in list(wm.memory_buffer)[-3:]:
            print(f"  • {item.content}")
            
        # RL learned values
        rl = self.council.modules['memory_rl']
        print(f"\n🧬 REINFORCEMENT LEARNING")
        print(f"  Total updates: {rl.total_updates}")
        print(f"  Learned locations: {len(rl.q_values)}")
        
        # Top valued actions
        if rl.q_values:
            top_values = sorted(
                [(s, a, v) for s, actions in rl.q_values.items() 
                 for a, v in actions.items()],
                key=lambda x: x[2], reverse=True
            )[:3]
            
            print("  Top valued actions:")
            for state, action, value in top_values:
                print(f"    {state} -> {action}: {value:.2f}")
