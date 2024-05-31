#!/usr/bin/env node
// emacs mode -*-JavaScript-*-

const fs = require('fs'),
      getopt = require('node-getopt')

// parse command-line options
const opt = getopt.create([
  ['m' , 'machine=PATH'     , 'path to JSON machine file'],
  ['s' , 'states=STRING'    , 'comma-separated list of states to include in the matrix'],
  ['z' , 'zerocols=STRING'  , 'comma-separated list of matrix columns to zero out'],
  ['e' , 'eliminate=STRING' , 'comma-separated list of states to eliminate'],
  ['p' , 'prefix=STRING'    , 'text to prefix matrix with'],
  ['y' , 'types'            , 'output state type sets'],
  ['t' , 'tex'              , 'output LaTex instead of Mathematica'],
  ['h' , 'help'             , 'display this help message']
])              // create Getopt instance
      .bindHelp()     // bind option 'help' to default action
      .parseSystem() // parse command line

if (!opt.options.machine)
  throw new Error ("please specify a machine")

const parseStateList = (s) => s.replaceAll(/(\d+)\.\.(\d+)/g,(_m,s,e)=>Array.from({length:parseInt(e)+1-parseInt(s)}).map((_,n)=>n+parseInt(s)).join(',')).split(',').map((s)=>parseInt(s))

const machine = JSON.parse (fs.readFileSync (opt.options.machine).toString())

let statesOfType = {};
const stateType = machine.state.map ((state) => {
  const id = state.id;
  let type;
  if (typeof(id) === 'string')
    type = id;
  else {
    const [fState, gState] = id;
    if (fState[0] == 'E' || gState[0] == 'E')
      type = 'E';
    else if (fState[0] == 'M' && gState[0] == 'M')
      type = 'M';
    else if (fState[0] == 'D')
      type = fState;
    else if (fState[0] == 'M' && (gState[0] == 'D' || gState[0] == 'I'))
      type = gState;
    else if (fState[0] == 'I' && gState[0] == 'D')
      type = 'N';
    else if (fState[0] == 'I' && gState[0] == 'M')
      type = fState;
    else if (fState[0] == 'I' && gState[0] == 'I')
      type = gState;
    else
      throw new Error('unknown state: ' + JSON.stringify(id))
  }
  (statesOfType[type] = statesOfType[type] || []).push(state.n);
  return type;
});

const states = (opt.options.states
		? parseStateList(opt.options.states)
		: stateType.map((_s,n)=>n)).filter((n)=>stateType[n] != 'E');

if (opt.options.eliminate) {
  const elimStates = parseStateList(opt.options.eliminate)
  let elim = {}
  elimStates.forEach ((s) => elim[s] = 1)
  machine.state.forEach ((state) => {
    if (state.trans)
      state.trans = state.trans.reduce ((list, t) => {
        return list.concat (elim[t.to]
                            ? machine.state[t.to].trans
                            : [t])
      }, [])
  })
}

const weight2str = (weight) => {
  const { str, scale } = weight2expr (weight)
  if (scale === 0)
    return scale
  if (scale === 1 && str === '')
    return scale
  return str + (typeof(scale) === 'undefined' || scale === 1 ? "" : (scale > 1 ? ("*" + scale) : ("/" + (1/scale))))
}

const weight2expr = (weight) => {
  if (typeof(weight) === 'string')
    return { str: weight, scale: 1 }
  if (typeof(weight) === 'number')
    return { str: '', scale: weight }
  if (weight["*"])
    return weight["*"].reduce ((expr, term) => {
      const termExpr = weight2expr (term)
      if (termExpr.str)
        expr.str = (expr.str ? (expr.str + "*") : "") + termExpr.str
      expr.scale *= termExpr.scale
      return expr
    }, { str: '', scale: 1 })
  if (weight["+"])
    return { str: "(" + weight["+"].map ((w) => weight2str(w)).join("+") + ")", scale: 1 }
  if (weight["-"])
    return { str: "(" + weight2str(weight["-"][0]) + "-(" + weight2str(weight["-"][1]) + "))", scale: 1 }
  if (weight.not)
    return { str: "(1-" + weight2str(weight.not) + ")", scale: 1 }
  throw new Error ("unknown weight expression: " + JSON.stringify(weight))
}

let matrix = states.map ((row) => states.map ((col) => {
  const state = machine.state[row]
  const trans = state.trans && state.trans.find ((trans) => trans.to === col)
  return weight2str (trans ? (trans.weight || 1) : 0)
}))

if (opt.options.zerocols)
  opt.options.zerocols.split(',').forEach ((col) => {
    matrix.forEach ((row) => row[col] = 0)
  })

if (opt.options.tex) {
  let tex = (opt.options.prefix || "") + matrix.map ((row) => row.join(" & ") + ' \\\\' + "\n").join("")
  tex = tex.replace(/lambda/g, "\\lambda")
  tex = tex.replace(/mu/g, "\\mu")
  tex = tex.replace(/dt/g, "\\Delta t")
  tex = tex.replace(/\*/g, " ")
  console.log (tex)
} else {
  const prefix = opt.options.prefix || "";
  console.log (prefix + "{" + matrix.map ((row) => "{" + row.join(",") + "}").join(",") + "}")
  if (opt.options.types) {
    const types = Object.keys(statesOfType).filter((t)=>t!='E').sort();
    types.forEach ((t) => {
      console.log (t + prefix + "{" + statesOfType[t].map((n)=>n+1).join(',') + "}");
    });
  }
}
