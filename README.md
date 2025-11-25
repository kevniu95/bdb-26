Note: This Big Data Bowl 2026 has two competitions. This competition is the Prediction competition. Learn more about the Analytics competition here.

The downfield pass is the crown jewel of American sports. When the ball is in the air, anything can happen, like a touchdown, an interception, or a contested catch. The uncertainty and the importance of the outcome of these plays is what helps keep audiences on the edge of its seat.

The 2026 Big Data Bowl is designed to help the National Football League better understand player movement during the pass play, starting with when the ball is thrown and ending when the ball is either caught or ruled incomplete. For the offensive team, this means focusing on the targeted receiver, whose job is to move towards the ball landing location in order to complete a catch. For the defensive team, who could have several players moving towards the ball, their jobs are to both prevent the offensive player from making a catch, while also going for the ball themselves. This year's Big Data Bowl asks our fans to help track the movement of these players.

In the Prediction Competition of the Big Data Bowl, participants are tasked with predicting player movement with the ball in the air. Specifically, the NFL is sharing data before the ball is thrown (including the Next Gen Stats tracking data), and stopping the play the moment the quarterback releases the ball. In addition to the pre-pass tracking data, we are providing participants with which offensive player was targeted (e.g, the targeted receiver) and the landing location of the pass.

Using the information above, participants should generate prediction models for player movement during the frames when the ball is in the air. The most accurate algorithms will be those whose output most closely matches the eventual player movement of each player.

Competition specifics

In the NFL's tracking data, there are 10 frames per second. As a result, if a ball is in the air for 2.5 seconds, there will be 25 frames of location data to predict.
Quick passes (less than half a second), deflected passes, and throwaway passes are dropped from the competition.
Evaluation for the training data is based on historical data. Evaluation for the leaderboard is based on data that hasn't happened yet. Specifically, we will be doing a live leaderboard covering the last five weeks of the 2025 NFL season.
Evaluation
Submissions are evaluated using the Root Mean Squared Error between the predicted and the observed target. The evaluation metric for this contest can be found here.


Submission File
You must submit to this competition using the provided evaluation API, which ensures that models perform inference on a single play at a time. For each row in the test dataframe you must predict the corresponding x and y values.