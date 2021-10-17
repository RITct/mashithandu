const MAX = 50;

let randomChoice = () => {
  return Math.floor(Math.random() * 3);
};
let choice = (number) => {
  return ["rock", "paper", "scissors"][number];
};
let score = ([firstChoice, secondChoice]) => {
  var dataBase = {
    rock: { rock: 0, paper: 0, scissors: 1 },
    paper: { rock: 1, paper: 0, scissors: 0 },
    scissors: { rock: 0, paper: 1, scissors: 0 },
  };
  var player1 = dataBase[firstChoice][secondChoice];
  var player2 = dataBase[secondChoice][firstChoice];
  return [player1, player2];
};

function final([p1, p2]) {
  if (p1 === 0) {
    return 0;
  } else if (p1 === 1) {
    return 1;
  }
}
let playerValue = () => {
  const p1 = choice(randomChoice());
  const p2 = choice(randomChoice());
  const p3 = choice(randomChoice());
  const p4 = choice(randomChoice());
  return [p1, p2, p3, p4];
};
let print_result = () => {
  let ans = [];
  players = ["player1", "player2", "player3", "player4"];
  let dict = {};

  for (let k = 0; k < MAX; k++) {
    //   console.log(dict);
    let a = playerValue();
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        if (players[i] != players[j]) {
          dict[`${players[i]} vs ${players[j]}`] = dict[
            `${players[i]} vs ${players[j]}`
          ]
            ? dict[`${players[i]} vs ${players[j]}`] +
              final(score([a[i], a[j]]))
            : final(score([a[i], a[j]]));
        }
      }
    }

    ans.push({ ...dict });
  }

  return ans;
};

const game = (req, res) => {
  const structure = "{'player wins vs against': 'score'}";
  res.json({ structure, results: print_result() });
};

module.exports = game;
