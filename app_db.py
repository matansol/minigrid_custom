from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import time

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///testdb.db'  # Uses a local SQLite database
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:GmGJtyAIzmnPuEjbUHFPBlTyxfFPvQOO@roundhouse.proxy.rlwy.net:22844/railway'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Optional: Disable modification tracking

# Create the SQLAlchemy db instance
db = SQLAlchemy(app)


def create_database():
    with app.app_context():
        db.create_all()


class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    actions = db.relationship('Action', backref='player', lazy=True)

class Action(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    action_type = db.Column(db.String(50))
    agent_action = db.Column(db.Boolean)
    score = db.Column(db.Float)
    reward = db.Column(db.Float)
    done = db.Column(db.Boolean)
    player_id = db.Column(db.Integer, db.ForeignKey('player.id'))
    episode = db.Column(db.Integer)
    timestamp = db.Column(db.Float)
    agent_index = db.Column(db.Integer)

class PlayerSession:
    def __init__(self, player_name):
        self.player_name = player_name
        player = Player.query.filter_by(name=player_name).first()
        if not player:
            player = Player(name=player_name)
            db.session.add(player)
            db.session.commit()
        self.player = player

    def record_action(self, action, score, reward, done, agent_action=False, episode=None, agent_index=None):
        new_action = Action(
            action_type=action,
            agent_action=agent_action,
            score=score,
            reward=reward,
            done=done,
            player_id=self.player.id,
            timestamp=time.time(),
            episode=episode,
            agent_index=agent_index
        )
        db.session.add(new_action)
        db.session.commit()
    
if __name__ == '__main__':
    create_database()