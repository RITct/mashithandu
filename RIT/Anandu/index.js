const express = require("express");
const game = require("./game");
const app = express();
app.use(express.json())
app.get("/game/start", game);

app.listen(3000, () => {
  console.log("server started at port 3000");
});
