import asyncio
import aiohttp
import socketio
import time
import random
import json
from datetime import datetime
import sys

class TutorialLoadTester:
    def __init__(self, server_url, num_users=5, connection_delay=1.0, action_interval=2.0):
        self.server_url = server_url
        self.num_users = num_users
        self.connection_delay = connection_delay
        self.action_interval = action_interval
        self.users = []
        self.stats = {
            'connected': 0,
            'playing': 0,
            'errors': 0,
            'total_actions': 0,
            'start_time': None
        }
        
    async def create_user(self, user_id):
        """Create a simulated user"""
        user = TutorialUser(user_id, self.server_url, self.action_interval, self.update_stats)
        self.users.append(user)
        return user
    
    def update_stats(self):
        """Update global statistics"""
        self.stats['connected'] = sum(1 for u in self.users if u.connected)
        self.stats['playing'] = sum(1 for u in self.users if u.playing)
        self.stats['errors'] = sum(u.error_count for u in self.users)
        self.stats['total_actions'] = sum(u.action_count for u in self.users)
    
    def print_stats(self):
        """Print current statistics"""
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        print(f"\n{'='*60}")
        print(f"LOAD TEST STATISTICS (Elapsed: {elapsed:.1f}s)")
        print(f"{'='*60}")
        print(f"Users Created: {len(self.users)}")
        print(f"Connected: {self.stats['connected']}")
        print(f"Playing: {self.stats['playing']}")
        print(f"Total Errors: {self.stats['errors']}")
        print(f"Total Actions: {self.stats['total_actions']}")
        print(f"Actions/sec: {self.stats['total_actions']/elapsed:.2f}" if elapsed > 0 else "Actions/sec: 0")
        print(f"{'='*60}")
    
    async def run_load_test(self, duration_seconds=60):
        """Run the load test for specified duration"""
        print(f"Starting load test with {self.num_users} users for {duration_seconds} seconds...")
        print(f"Server: {self.server_url}")
        
        self.stats['start_time'] = time.time()
        
        # Create and connect users with staggered timing
        tasks = []
        for i in range(self.num_users):
            task = asyncio.create_task(self.simulate_user_lifecycle(i + 1))
            tasks.append(task)
            await asyncio.sleep(self.connection_delay)
        
        # Monitor progress
        monitor_task = asyncio.create_task(self.monitor_progress(duration_seconds))
        tasks.append(monitor_task)
        
        try:
            # Wait for all tasks to complete or timeout
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 
                                 timeout=duration_seconds + 10)
        except asyncio.TimeoutError:
            print("Load test completed (timeout)")
        except KeyboardInterrupt:
            print("\nLoad test interrupted by user")
        finally:
            # Cleanup
            for user in self.users:
                await user.disconnect()
            
            # Final stats
            self.print_stats()
            print("\nLoad test completed!")
    
    async def simulate_user_lifecycle(self, user_id):
        """Simulate a complete user session"""
        user = await self.create_user(user_id)
        
        try:
            await user.connect()
            await asyncio.sleep(1)  # Wait a bit after connection
            
            if user.connected:
                await user.start_game()
                await asyncio.sleep(1)  # Wait for game to start
                
                # Play for a while
                await user.play_game()
                
        except Exception as e:
            print(f"User {user_id} encountered error: {e}")
            user.error_count += 1
        finally:
            await user.disconnect()
    
    async def monitor_progress(self, duration_seconds):
        """Monitor and print progress during test"""
        start_time = time.time()
        last_print = 0
        
        while time.time() - start_time < duration_seconds:
            await asyncio.sleep(5)  # Print stats every 5 seconds
            
            current_time = time.time() - start_time
            if current_time - last_print >= 5:
                self.update_stats()
                self.print_stats()
                last_print = current_time


class TutorialUser:
    def __init__(self, user_id, server_url, action_interval, stats_callback):
        self.user_id = user_id
        self.server_url = server_url
        self.action_interval = action_interval
        self.stats_callback = stats_callback
        self.sio = None
        self.connected = False
        self.playing = False
        self.started = False  # whether start_game was emitted for the current connection
        self.error_count = 0
        self.action_count = 0
        self.actions = ['ArrowUp', 'ArrowLeft', 'ArrowRight', 'PageUp']
        
    async def connect(self):
        """Connect to the server"""
        try:
            # Prefer WebSocket only to avoid long-polling issues behind Azure reverse proxies
            self.sio = socketio.AsyncClient(
                reconnection=True,
                reconnection_attempts=3,
                reconnection_delay=1,
                reconnection_delay_max=5,
                logger=False,
                engineio_logger=False
            )
            
            # Event handlers
            @self.sio.event
            async def connect():
                self.connected = True
                self.log("Connected successfully")
                # On every (re)connect, emit start_game once for this connection
                if not self.started:
                    # slight delay to allow connection to settle
                    await asyncio.sleep(0.1)
                    await self.start_game()
                self.stats_callback()
            
            @self.sio.event
            async def connect_error(data):
                self.error_count += 1
                self.log(f"Connection error: {data}")
                self.stats_callback()
            
            @self.sio.event
            async def disconnect():
                self.connected = False
                self.playing = False
                self.started = False  # require a new start_game on reconnect
                self.log("Disconnected")
                self.stats_callback()
            
            @self.sio.event
            async def game_update(data):
                if not self.playing:
                    self.playing = True
                    self.log("Game started - beginning actions")
                    self.stats_callback()
                # self.log(f"Game update - Score: {data.get('score', 0)}, Steps: {data.get('steps', 0)}")
            
            @self.sio.event
            async def episode_finished(data):
                self.log(f"Episode finished - Score: {data.get('score', 0)}")
                # Start next episode after a short delay
                await asyncio.sleep(1)
                if self.connected:
                    await self.sio.emit('next_episode')
            
            @self.sio.event
            async def error(data):
                self.error_count += 1
                self.log(f"Server error: {data.get('error', 'Unknown error')}")
                self.stats_callback()
            
            # Connect to server
            await self.sio.connect(self.server_url, transports=['polling', 'websocket'])
            
        except Exception as e:
            self.error_count += 1
            self.log(f"Failed to connect: {e}")
            self.stats_callback()
    
    async def start_game(self):
        """Start the game"""
        if self.sio and self.connected:
            try:
                await self.sio.emit('start_game', {
                    'playerName': f'LoadTest_User_{self.user_id}',
                    'finalStep': 0
                })
                self.started = True
                self.log("Game start request sent")
            except Exception as e:
                self.error_count += 1
                self.log(f"Failed to start game: {e}")
                self.stats_callback()
    
    async def play_game(self):
        """Simulate playing the game"""
        while self.connected and self.playing:
            try:
                # Send random action
                action = random.choice(self.actions)
                await self.sio.emit('send_action', {
                    'action': action,
                    'episode_num': 1
                })
                self.action_count += 1
                
                # Wait before next action
                await asyncio.sleep(self.action_interval + random.uniform(-0.5, 0.5))
                
            except Exception as e:
                self.error_count += 1
                self.log(f"Error sending action: {e}")
                self.stats_callback()
                break
    
    async def disconnect(self):
        """Disconnect from server"""
        self.playing = False
        if self.sio and self.connected:
            try:
                await self.sio.disconnect()
            except:
                pass
        self.connected = False
        self.stats_callback()
    
    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] User {self.user_id}: {message}")


async def main():
    """Main function to run load test"""
    if len(sys.argv) < 2:
        print("Usage: python load_test.py <server_url> [num_users] [duration_seconds]")
        print("Example: python load_test.py https://dpu-tutorial-app.yellowmushroom-27e70244.westeurope.azurecontainerapps.io 10 60")
        return
    
    server_url = sys.argv[1]
    num_users = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    duration_seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 60
    
    # Create and run load tester
    tester = TutorialLoadTester(
        server_url=server_url,
        num_users=num_users,
        connection_delay=1.0,  # 1 second between user connections
        action_interval=2.0    # 2 seconds between actions
    )
    
    await tester.run_load_test(duration_seconds)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nLoad test interrupted by user")
