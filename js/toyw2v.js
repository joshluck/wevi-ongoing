$(document).ready(function() {
  set_default_training_data();
  set_default_config();
  global_init();

  $("#btn-restart").click(updateAndRestartButtonClick);
  $("#btn-next").click(nextButtonClick);
  //$("#btn-pca").click(updatePCAButtonClick);
  $("#btn-next20").click(function(){batchTrain(20)});
  $("#btn-next100").click(function(){batchTrain(100)});
  $("#btn-next500").click(function(){batchTrain(500)});
  $("#btn-learning-rate").click(function(){load_config()});
  $("#btn-restart-all").click(updateAndRestartButtonClick);
  $("#btn-create-network").click(generate_network);
});

function global_init() {
  modal_active = 1; // remove after debugging

  $("#error").empty().hide();
  load_training_data();  // this needs to be loaded first to determine vocab

  load_config(network_array[current_network_index]);

  for(var i = 0; i < network_array.length; i++) {
    setup_neural_net(network_array[i]);  
    create_huffman(network_array[i]);
  }
    
  JSONtree = tree_to_JSON(network_array[0].hufftree.root);
  setup_neural_net_svg();

  for(var i = 0; i < network_array.length; i++) {
    update_neural_net_svg(network_array[i]);
  }

  setup_neural_net_svg();
  update_neural_net_svg(network_array[current_network_index]);
  
  setup_heatmap_svg();
  update_heatmap_svg();

  for(var i = 0; i < network_array.length; i++) {
    setup_output(network_array[i]);
    update_output_svg(network_array[i]);
  }

  // initial feed-forward
  for(var i = 0; i < network_array.length; i++) {
    do_feed_forward(network_array[i]);
  }

  update_neural_excite_value();
  update_id(network_array[current_network_index]);
}

function set_default_config() {
  num_active_networks = 0; //global
  network_array = []; //global
  current_network_index = 0; //global
  modal_active = 0; //global
  excite_flag = false;

  //generate_network();

  var preset_HS = [{name: "Yes"}, {name: "No"},];
  var preset_random_state = [{val: 1}, {val: 2}, {val: 3},];
  var preset_learning_rate = [{val: 0.1}, {val: 0.2}, {val: 0.3}, {val: 0.4}, {val: 0.5},];

  // first do HS defaults
  var select = d3.select("#sel-HS");
  var options = select.selectAll("option")
    .data(preset_HS)
    .enter()
    .append("option")
    .attr("value", function(d) {return d.name})
    .html(function(d) {return d.name});
  
  select.on("change", function() {
    updateAndRestartButtonClick();
  });

  // now do random_state defaults
  select = d3.select("#sel-random");
  options = select.selectAll("option")
    .data(preset_random_state)
    .enter()
    .append("option")
    .attr("value", function(d) {return d.val})
    .html(function(d) {return d.val});

  select.on("change", function() {
    updateAndRestartButtonClick();
  });

  // now do learning rate defaults
  select = d3.select("#sel-learning-rate");
  options = select.selectAll("option")
    .data(preset_learning_rate)
    .enter()
    .append("option")
    .attr("value", function(d) {return d.val})
    .html(function(d) {return d.val});

  select.on("change", function() {
    updateAndRestartButtonClick();
  }); 
}

function update_layer_presets(vocab_size, network) {
  var preset_layers = [];
  
  for(var i = 0; i < 5; i++) {
    preset_layers.push(i + 1);
  }

  var select = d3.select("#sel-layers");

  var options = select.selectAll("option")
    .data(preset_layers)
    .enter()
    .append("option")
    .attr("value", function(d) {return d})
    .html(function(d) {return d});

  network.layer_count = (1 + select.property("selectedIndex"));

  hiddenVector = [];
  for(var i = 0; i < network_array[current_network_index].hidden_size; i++) {
    hiddenVector.push(0);
  }

  select.on("change", function() {
    updateAndRestartButtonClick();
  });
}

function increase_neurons_in_layer(vocab_size, network, layer) {
  if(network.hiddenNeurons[layer].neuron_count < vocab_size) {
    ++network.hiddenNeurons[layer].neuron_count;

    var j = network.hiddenNeurons[layer].neuron_count;
    network.hiddenNeurons[layer].data.push({value: 0, idx: j});

    if((layer + 1) == network.layer_count) {
      network.hidden_size++;
    }
  }

  hiddenVector = [];
  for(var i = 0; i < network_array[current_network_index].hidden_size; i++) {
    hiddenVector.push(0);
  }

  updateAndRestartButtonClick();
}

function decrease_neurons_in_layer(network, layer) {
  if(network.hiddenNeurons[layer].neuron_count > 1) {
    --network.hiddenNeurons[layer].neuron_count;

    network.hiddenNeurons[layer].data.pop();

    if((layer + 1) == network.layer_count) {
      --network.hidden_size;
    }
  }

  hiddenVector = [];
  for(var i = 0; i < network_array[current_network_index].hidden_size; i++) {
    hiddenVector.push(0);
  }

  updateAndRestartButtonClick();
}

function load_config(network) {
  network.use_hs = (1 - d3.select('#sel-HS').property("selectedIndex"));
  network.random_state = (1 + d3.select('#sel-random').property("selectedIndex"));
  network.learning_rate = 0.1 * (1 + d3.select('#sel-learning-rate').property("selectedIndex"));

  update_layer_presets(vocab.length, network);
  update_id(network);
}

function update_id(network) {
  $('#network-id' + network.id).empty()
    .append("Network " + network.id);
  $('#network-hs-id' + network.id).empty()
    .append(network.use_hs);
  $('#network-random-id' + network.id).empty()
    .append(network.random_state);
  $('#network-learningrate-id' + network.id).empty()
    .append(network.learning_rate);
  $('#network-layer-id' + network.id).empty()
    .append(network.layer_count);
}

function update_hidden_size(network) {
  network.hidden_size = network.hiddenNeurons[network.hiddenNeurons.length - 1].neuron_count;
}

function local_update(network) {
  // only do if more than one neural network
  if(num_active_networks > 1) {
    setup_neural_net_svg();
    update_neural_net_svg(network);

    setup_heatmap_svg();
    update_heatmap_svg();

    for(var i = 0; i < network_array.length; i++) {
      update_hidden_size(network_array[i]);
      setup_output(network_array[i]);
      update_output_svg(network_array[i]); 
    }

    update_neural_excite_value();
  }
}

function switch_active_network(network) {
  current_network_index = network.id - 1;

  $('#network-id-header').empty()
    .append("Network " + network.id + " Control Panel");

  for(var i = 0; i < network_array.length; i++) {
    $('#network-body-id' + network_array[i].id).css("background", "white");
  }
  
  $('#network-body-id' + network.id).css("background", "lightblue");

  // switch panel to reflect new network
  $('#sel-random').val(network.random_state);
  $('#sel-learning-rate').val(network.learning_rate);
  if(network.use_hs) {
    $('#sel-HS').val("Yes");
  } else {
    $('#sel-HS').val("No");
  }
  $('#sel-layers').val(network.layer_count);

  // only do if more than one neural network
  local_update(network);
}

function set_default_training_data() {
  var presets = 
    [{name:"Fruit and juice", data:"eat|apple,eat|orange,eat|rice,drink|juice,drink|milk,drink|water,orange|juice,apple|juice,rice|milk,milk|drink,water|drink,juice|drink"},
     {name:"Fruit and juice (CBOW)", data: "drink^juice|apple,eat^apple|orange,drink^juice|rice,drink^milk|juice,drink^rice|milk,drink^milk|water,orange^apple|juice,apple^drink|juice,rice^drink|milk,milk^water|drink,water^juice|drink,juice^water|drink"},
     {name:"Fruit and juice (Skip-gram)", data: "apple|drink^juice,orange|eat^apple,rice|drink^juice,juice|drink^milk,milk|drink^rice,water|drink^milk,juice|orange^apple,juice|apple^drink,milk|rice^drink,drink|milk^water,drink|water^juice,drink|juice^water"},
     {name:"Self loop (5-point)", data:"A|A,B|B,C|C,D|D,E|E"},
     {name:"Directed loop (5-point)", data:"A|B,B|C,C|D,D|E,E|A"},
     {name:"Undirected loop (5-point)", data:"A|B,B|C,C|D,D|E,E|A,B|A,C|B,D|C,E|D,A|E"},
     {name:"King and queen", data: "king|kindom,queen|kindom,king|palace,queen|palace,king|royal,queen|royal,king|George,queen|Mary,man|rice,woman|rice,man|farmer,woman|farmer,man|house,woman|house,man|George,woman|Mary"},
     {name:"King and queen (symbol)", data: "king|a,queen|a,king|b,queen|b,king|c,queen|c,king|x,queen|y,man|d,woman|d,man|e,woman|e,man|f,woman|f,man|x,woman|y"},
    ];
  
  $('#input-text').html(presets[0].data);

  var select = d3.select("#sel-presets");

  var options = select.selectAll("option")
    .data(presets)
    .enter()
    .append("option")
    .attr("value", function(d) {return d.name})
    .html(function(d) {return d.name});

  select.on("change", function() {
    var selectedIndex = select.property("selectedIndex");
    var selectedPreset = options.filter(function(d,i) {return i == selectedIndex});
    $('#input-text').html(selectedPreset.datum().data);
    updateAndRestartButtonClick();
  });
}

function load_training_data() {
  input_pairs = [];  // global
  non_unique_vocab = []; //global
  vocab = [];  // global
  current_input = null;  // global, when inactivate, should be null
  current_input_idx = -1;
  var input_text = $("#input-text").val();
  var pairs = input_text.trim().split(",")
  pairs.forEach(function(s) {
    tokens = s.trim().split("|");
    assert(tokens.length == 2, "Bad input format: " + s);
    tokens[0] = tokens[0].trim().split("^");  // input tokens
    tokens[1] = tokens[1].trim().split("^");  // output tokens
    input_pairs.push(tokens);
    tokens[0].forEach(function(t) {vocab.push(t)});
    tokens[1].forEach(function(t) {vocab.push(t)});
    tokens[0].forEach(function(t) {non_unique_vocab.push(t)});
    tokens[1].forEach(function(t) {non_unique_vocab.push(t)});
  });
  non_unique_vocab.sort();
  vocab = $.unique(vocab.sort());

  expected_word = ""; //global

  if(!num_active_networks) {generate_network();}
}

// NETWORK OBJECT SECTION //

function Network(id, layers, hs, random, learning) {
  this.id = id;
  this.layer_count = layers;
  this.hidden_size = layers;
  this.use_hs = hs;
  this.random_state = random;
  this.learning_rate = learning;
}

function generate_network() {
  if(num_active_networks > 2) {return;}
  
  num_active_networks++;

  network_array.push(new Network(num_active_networks, 1, 1, 1, 0.1));

  $('#insert-network' + num_active_networks).after('<div class="col-sm-1 nopadding" id="insert-network' + (num_active_networks + 1) + '">'
    //+ '<div class="panel panel-default">'
    //+ '<div class="panel-body" data-parent="#accordion" data-toggle="collapse" data-target="#collapseNetworkOne">'
    + '<div class="panel panel-default">'
    + '<div class="panel-heading" id="network-id' + num_active_networks + '"></div>'
    + '<a href="#" onclick = "switch_active_network(network_array[' + (num_active_networks - 1) + '])">'
    + '<div class="panel-body" id="network-body-id' + num_active_networks + '">'
    + '<p id="network-random-id' + num_active_networks + '" style="text-indent: 2em;"></p>'
    + '<p id="network-learningrate-id' + num_active_networks + '" style="text-indent: 2em;"></p>'    
    + '<p id="network-hs-id' + num_active_networks + '" style="text-indent: 2em;"></p>'
    + '<p id="network-layer-id' + num_active_networks + '" style="text-indent: 2em;"></p>'
    + '</div>'
    + '<div id="output-vis' + num_active_networks + '"></div>'
    + '</a>'
    + '</div>'
    //+ '</div>'
    //+ '</div>'
    + '</div>');

  setup_neural_net(network_array[num_active_networks - 1]);
  create_huffman(network_array[num_active_networks - 1]);
  switch_active_network(network_array[num_active_networks - 1]);

  global_init();
}

// BINARY TREE SECTION //

function Node(val) {
  this.value = val;
  this.gradient = [];
  this.recent_gradient = [];
  this.parent = null;
  this.left = null;
  this.right = null;
  this.vect = [];
  this.index = 0;
}

function HuffmanTree() {
  this.root = null;
  this.data = [];
}

function create_huffman(network) {
  //create frequency table
  huff_words = [];
  huff_counts = [];
  var prev;

  for (var i = 0; i < non_unique_vocab.length; i++) {
    if (non_unique_vocab[i] !== prev) {
      huff_words.push(non_unique_vocab[i]);
      huff_counts.push(1);
    } else {
      huff_counts[huff_counts.length - 1]++;
    }
    prev = non_unique_vocab[i];
  }

  //create huffman tree
  network.hufftree = new HuffmanTree();
  huff = huff_words.map(function(e, i) {
    return [e, huff_counts[i]];
  });
  
  while(huff.length > 1) {
    huff.sort(function(x,y) {return x[1] - y[1]});

    var temp1 = huff[0];
    var temp2 = huff[1];
    huff.shift();
    huff.shift();

    if (temp1[0].indexOf('treetemp') !== -1) { //if temp1 IS NOT an actual leaf
      var pos1 = parseInt(temp1[0].replace(/[^0-9\.]/g, ''), 10);
      var tempnode1 = network.hufftree.data[pos1];

      if (temp2[0].indexOf('treetemp') == -1) { //if temp2 IS an actual leaf
        var tempnode2 = new Node(temp2[0]);
        var tempparent = new Node('treetemp' + network.hufftree.data.length);

        tempparent.left = tempnode1;
        tempparent.right = tempnode2;
        tempnode1.parent = tempparent;
        tempnode2.parent = tempparent;

        network.hufftree.data[pos1] = tempnode1;
        network.hufftree.data.push(tempnode2);
        network.hufftree.data.push(tempparent);
      } else { //if temp2 IS NOT an actual leaf
        var pos2 = parseInt(temp2[0].replace(/[^0-9\.]/g, ''), 10);
        var tempnode2 = network.hufftree.data[pos2];
        var tempparent = new Node('treetemp' + network.hufftree.data.length);

        tempparent.left = tempnode1;
        tempparent.right = tempnode2;
        tempnode1.parent = tempparent;
        tempnode2.parent = tempparent;

        network.hufftree.data[pos1] = tempnode1;
        network.hufftree.data[pos2] = tempnode2;
        network.hufftree.data.push(tempparent);
      }
    } else { //if temp1 IS an actual leaf
      var tempnode1 = new Node(temp1[0]);

      if (temp2[0].indexOf('treetemp') == -1) { //if temp2 IS an actual leaf
        var tempnode2 = new Node(temp2[0]);
        var tempparent = new Node('treetemp' + network.hufftree.data.length);

        tempparent.left = tempnode1;
        tempparent.right = tempnode2;
        tempnode1.parent = tempparent;
        tempnode2.parent = tempparent;

        network.hufftree.data.push(tempnode1);
        network.hufftree.data.push(tempnode2);
        network.hufftree.data.push(tempparent);
      } else { //if temp2 IS NOT an actual leaf
        var pos2 = parseInt(temp2[0].replace(/[^0-9\.]/g, ''), 10);
        var tempnode2 = network.hufftree.data[pos2];
        var tempparent = new Node('treetemp' + network.hufftree.data.length);

        tempparent.left = tempnode1;
        tempparent.right = tempnode2;
        tempnode1.parent = tempparent;
        tempnode2.parent = tempparent;

        network.hufftree.data.push(tempnode1);
        network.hufftree.data[pos2] = tempnode2;
        network.hufftree.data.push(tempparent);
      }
    }

    huff.unshift(['treetemp' + (network.hufftree.data.length - 1), temp1[1] + temp2[1]]); //replace with NON-LEAF node
  }
  
  network.outputNodes = [];
  network.hufftree.root = network.hufftree.data[network.hufftree.data.length - 1];

  network.hufftree.data.forEach(function(n, i) {
    for (var j = 0; j < network.hidden_size; j++) {
      n.vect.push(get_random_init_weight(network.hidden_size));
      n.gradient.push(0.0);
      n.recent_gradient.push(0.0);
    }

    n.index = i;

    network.outputNodes.push(n);
  });
  
}

function tree_to_JSON(w) { //MAKE SURE to call on hufftree.root
  var json_in_progress = "";

  if (w.value.indexOf("treetemp") == -1) {
    json_in_progress += ("{\n\t \"name\": \"" + w.value + "\",\n");  
  } else {
    json_in_progress += ("{\n\t \"name\": \"\", \n");
  }
  json_in_progress += ("\t \"parent\": \"null\"");
  
  if ((w.left == null) && (w.right == null) && (w == w.parent.left)) {
    json_in_progress += "\n},\n";
    return json_in_progress;
  } else if ((w.left == null) && (w.right == null)) {
    json_in_progress += "\n}\n";
    //json_in_progress += "]\n"
    return json_in_progress;
  }

  json_in_progress += (",\n\t \"children\": [\n" + tree_to_JSON(w.left) + tree_to_JSON(w.right) + "]\n}");
  if (w.parent != null && w == w.parent.left) {json_in_progress += ",\n"}
  return json_in_progress;
}

// END OF BINARY TREE SECTION //

// "context word" === "input word"
function isCurrentContextWord(w) {
  if (current_input == null) return;
  var context_words = current_input[0];
  if (context_words.length == 1) {
    return w == context_words[0];
  } else if (context_words.length > 1) {
    var matched = false;
    context_words.forEach(function(cw) {
      if (cw == w) {
        matched = true;
        return;
      }
    });
    return matched;
  }
  return false;
}

function isCurrentTargetWord(w) {
  if (current_input == null) return;
  var target_words = current_input[1];
  if (target_words.length == 1) {
    return w == target_words[0];
  } else if (target_words.length > 1) {
    var matched = false;
    target_words.forEach(function(tw) {
      if (tw == w) {
        matched = true;
        return;
      }
    });
    return matched;
  }
  return false;
}

// Regardless the value of current_input, forward to the next input
function activateNextInput() {
  current_input_idx = (current_input_idx + 1) % input_pairs.length;
  current_input = input_pairs[current_input_idx];
  
  for(var i = 0; i < network_array.length; i++) {
    network_array[i].inputNeurons.forEach(function(n, i) {
      n['value'] = isCurrentContextWord(n['word']) ? 1 : 0;
      n['always_excited'] = isCurrentContextWord(n['word']);
    });
    
    do_feed_forward(network_array[i]);  // model  
  }
  
  update_neural_excite_value();  // visual
}

function deactivateCurrentInput() {
  current_input = null;

  for(var i = 0; i < network_array.length; i++) {
    network_array[i].inputNeurons.forEach(function(n, i) {
      // n['value'] = 0;
      n['always_excited'] = false;
    });
    
    do_feed_forward(network_array[i]);  // model  
  }
  
  update_neural_excite_value();  // visual
}



// SPECIAL ACTIVATE/DEACTIVATE for BATCH TRAINING AND ANIMATION

function activateNextInput_modified() {
  current_input_idx = (current_input_idx + 1) % input_pairs.length;
  current_input = input_pairs[current_input_idx];

  for(var i = 0; i < network_array.length; i++) {
    network_array[i].inputNeurons.forEach(function(n, i) {
      n['value'] = isCurrentContextWord(n['word']) ? 1 : 0;
      n['always_excited'] = isCurrentContextWord(n['word']);
    });
  }

  for(var i = 0; i < network_array.length; i++) {
    do_feed_forward(network_array[i]);  // model
  }

  update_neural_excite_value();  // visual
}

function deactivateCurrentInput_modified() {
  current_input = null;

  for(var i = 0; i < network_array.length; i++) {
    network_array[i].inputNeurons.forEach(function(n, i) {
      // n['value'] = 0;
      n['always_excited'] = false;
    });
  }

  for(var i = 0; i < network_array.length; i++) {
    do_feed_forward(network_array[i]);  // model
  }

  update_neural_excite_value();  // visual
}

// END OF MODIFICATIONS



function HiddenNeuronLayer(val) {
  this.neuron_count = val;
  this.data = [];
}

function show_error(e) {
  console.log(e);
  var new_error = '<p>' + e + '</p>';
  $('#error').append(new_error);
  $('#error').show();
}

function setup_neural_net(network) {
  network.inputNeurons = [];  // global (same below)
  network.outputNeurons = [];
  vocab.forEach(function(word, i) {
    network.inputNeurons.push({word: word, value: 0, idx: i});
    network.outputNeurons.push({word: word, value: 0, idx: i});
  });

  if(network.hiddenNeurons != undefined) {
    var tempneuroncounts = [];
    network.hiddenNeurons.forEach(function(n, i) {tempneuroncounts.push(n.neuron_count);})
  } else {
    var tempneuroncounts = [];
    for (var i = 0; i < network.layer_count; i++) {tempneuroncounts.push(0);}
  }

  network.hiddenNeurons = [];

  for (var i = 0; i < network.layer_count; i++) {
    if(tempneuroncounts[i] != undefined) {network.hiddenNeurons.push(new HiddenNeuronLayer(tempneuroncounts[i]));}
    else {network.hiddenNeurons.push(new HiddenNeuronLayer(0));}

    if(network.hiddenNeurons[i].neuron_count == 0) {network.hiddenNeurons[i].neuron_count++;}

    for (var j = 0; j < network.hiddenNeurons[i].neuron_count; j++) {
      network.hiddenNeurons[i].data.push({value: 0, idx: j});
    }
  }

  network.hidden_size = network.hiddenNeurons[network.layer_count - 1].neuron_count;

  vocabSize = vocab.length;
  network.inputEdges = [];
  network.outputEdges = [];
  network.inputVectors = [];  // keeps references to the same set of underlying objects as inputEdges
  network.outputVectors = [];
  network.hiddenEdges = [];
  network.hiddenVectors = [];
  network.outputNodes = []; // keeps references to hufftree.data, essentially
  network.perplexity = [];
  seed_random(network.random_state);  // vector_math.js
  for (var i = 0; i < vocabSize; i++) {
    var inVecTmp = [];
    var outVecTmp = [];
    for (var j = 0; j < network.hiddenNeurons[0].neuron_count; j++) {
      var inWeightTmp = {source: i, target: j, weight: get_random_init_weight(network.hiddenNeurons[0].neuron_count)};
      network.inputEdges.push(inWeightTmp);
      inVecTmp.push(inWeightTmp);
    }
    for (var j = 0; j < network.hidden_size; j++) {
      var outWeightTmp = {source: j, target: i, weight: get_random_init_weight(network.hidden_size)};
      network.outputEdges.push(outWeightTmp);
      outVecTmp.push(outWeightTmp);
    }
    network.inputVectors.push(inVecTmp);
    network.outputVectors.push(outVecTmp);
  }
  for (var i = 0; i < network.layer_count - 1; i++) {
    network.hiddenVectors.push([]);
    for (var j = 0; j < network.hiddenNeurons[i].neuron_count; j++) {
      var hidVecTmp = [];
      for (var k = 0; k < network.hiddenNeurons[i + 1].neuron_count; k++) {
        var hidWeightTmp = {source: j, target: k, weight: get_random_init_weight(network.hiddenNeurons[i].neuron_count)};
        network.hiddenEdges.push(hidWeightTmp);
        hidVecTmp.push(hidWeightTmp);
      }
      network.hiddenVectors[i].push(hidVecTmp);
    }
  }
}

function setup_neural_net_svg() {
  nn_svg_width = 1000;  // view box, not physical
  nn_svg_height = 600;  // W/H ratio should match padding-bottom in wevi.css
  d3.select('div#neuron-vis > *').remove();
  nn_svg = d3.select('div#neuron-vis')
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .classed("neural-net", true)
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMinYMin meet")
   .attr("viewBox", "0 0 " + nn_svg_width + " " + nn_svg_height)
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .classed("neural-net", true);  // for picking up svg from outside

   /* Adding a colored svg background to help debug. */
  // svg.append("rect")
  //   .attr("width", "100%")
  //   .attr("height", "100%")
  //   .attr("fill", "#E8E8EE");

  // Prepare for drawing arrows indicating inputs/outputs
  nn_svg.append('svg:defs')
    .append("svg:marker")
    .attr("id", "marker_arrow")
    .attr('markerHeight', 3.5)
    .attr('markerWidth', 5)
    .attr('markerUnits', 'strokeWidth')
    .attr('orient', 'auto')
    .attr('refX', 0)
    .attr('refY', 0)
    .attr('viewBox', '-5 -5 10 10')
    .append('svg:path')
      .attr('d', 'M 0,0 m -5,-5 L 5,0 L -5,5 Z')
      .attr('fill', io_arrow_color());
}

function update_neural_net_svg(network) {
  var colors = ["#427DA8", "#6998BB", "#91B3CD", "#BAD0E0", 
                "#E1ECF3", "#FADEE0", "#F2B5BA", "#EA8B92", 
                "#E2636C", "#DB3B47"];
  numToColor = d3.scaleLinear()
    .domain(d3.range(0, 1, 1 / (colors.length - 1)))
    .range(colors);  // global

  var inputNeuronCX = nn_svg_width * 0.1;
  var outputNeuronCX = nn_svg_width - inputNeuronCX;
  var ioNeuronCYMin = nn_svg_height * 0.125;
  var ioNeuronCYInt = (nn_svg_height - 2 * ioNeuronCYMin) / (vocabSize - 1 + 1e-6);
  var hiddenNeuronCX = nn_svg_width / (network.layer_count + 1.5);
  var hiddenNeuronCXInt = (nn_svg_width - 2 * hiddenNeuronCX) / (network.layer_count + 1.75 + 1e-6);
  var hiddenNeuronCYMin = nn_svg_height * 0.15;
  var hiddenNeuronCYIntInput = (nn_svg_height - 2 * hiddenNeuronCYMin) / (network.hiddenNeurons[0].neuron_count - 1 + 1e-6);
  var hiddenNeuronCYIntOutput = (nn_svg_height - 2 * hiddenNeuronCYMin) / (network.hidden_size - 1 + 1e-6);
  var neuronRadius = nn_svg_width * 0.015;
  var neuronLabelOffset = neuronRadius * 1.4;

  if(network.use_hs) {
    draw_tree_interface();
  }

  var inputNeuronElems = nn_svg
    .selectAll("g.input-neuron")
    .data(network.inputNeurons)
    .enter()
    .append("g")
    .classed("input-neuron", true)
    .classed("neuron", true);

  inputNeuronElems
    .append("circle")
    .attr("cx", inputNeuronCX)
    .attr("cy", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * i});

  inputNeuronElems
    .append("text")
    .classed("neuron-label", true)
    .attr("x", inputNeuronCX - neuronLabelOffset)
    .attr("y", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * i})
    .attr("text-anchor", "end");

  if(network.use_hs) {
    d3.selectAll(".node--leaf")
      .classed("output-neuron", true)
      .classed("neuron", true);

    network.outputNeuronElems = d3.selectAll(".node--leaf")
      .datum(function(d) { return {word: d3.select(this).attr("word")}; });

    network.outputNeuronElems.data(network.outputNeurons, function(d) {return d['word'];});

    if(network.id == current_network_index + 1) {
      network.outputNeuronElems
        .append("circle");
    }
  } else {
    network.outputNeuronElems = nn_svg
      .selectAll("g.output-neuron")
      .data(network.outputNeurons)
      .enter()
      .append("g")
      .classed("output-neuron", true)
      .classed("neuron", true);

    if(network.id == current_network_index + 1) {
      network.outputNeuronElems
        .append("circle")
        .attr("cx", outputNeuronCX)
        .attr("cy", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * i});

      network.outputNeuronElems
        .append("text")
        .classed("neuron-label", true)
        .attr("x", outputNeuronCX + neuronLabelOffset)
        .attr("y", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * i})
        .attr("text-anchor", "start");
    }   
  }
  

  for(var j = 0; j < network.layer_count; j++) {
    var hiddenNeuronCYInt = (nn_svg_height - 2 * hiddenNeuronCYMin) / (network.hiddenNeurons[j].neuron_count - 1 + 1e-6);

    nn_svg.selectAll("g.hidden-neuron" + j)
    .data(network.hiddenNeurons[j].data)
    .enter()
    .append("g")
    .classed("hidden-neuron" + j, true)
    .classed("neuron", true)
    .append("circle")
    .attr("cx", hiddenNeuronCX + (j * hiddenNeuronCXInt))
    .attr("cy", function (d, i) {return hiddenNeuronCYMin + hiddenNeuronCYInt * i;});  

    $('<img src="plus.png" id="hidden-inc-' + j + '"'
      + 'onclick="increase_neurons_in_layer(' + vocabSize + ',' + 'network_array[' + current_network_index + '], ' + j + ')"'
      + 'style="position:absolute; top:' + (hiddenNeuronCYMin - 70) + 'px; left:' + (hiddenNeuronCX + j * hiddenNeuronCXInt) * 0.7 + 'px; width:20px; height:20px;">')
      .appendTo('#neuron-vis > div');

    $('<img src="minus.png" id="hidden-inc-' + j + '"'
      + 'onclick="decrease_neurons_in_layer(' + 'network_array[' + current_network_index + '], ' + j + ')"'
      + 'style="position:absolute; top:' + (hiddenNeuronCYMin - 70) + 'px; left:' + ((hiddenNeuronCX + j * hiddenNeuronCXInt) * 0.7 - 30) + 'px; width:20px; height:20px;">')
      .appendTo('#neuron-vis > div');
  }


  d3.selectAll("g.neuron > circle")
    .attr("r", neuronRadius)
    .attr("stroke-width", "2")
    .attr("stroke", "grey")
    .attr("fill", function(d) {return numToColor(0.5);});

  nn_svg.selectAll(".neuron-label")
    .attr("alignment-baseline", "middle")
    .style("font-size", 24)
    .text(function(d) {return d.word});

  nn_svg.selectAll("g.input-edge")
    .data(network.inputEdges)
    .enter()
    .append("g")
    .classed("input-edge", true)
    .classed("edge", true)
    .append("line")
    .attr("x1", inputNeuronCX + neuronRadius)
    .attr("x2", hiddenNeuronCX - neuronRadius)
    .attr("y1", function (d) {return ioNeuronCYMin + ioNeuronCYInt * d['source']})
    .attr("y2", function (d) {return hiddenNeuronCYMin + hiddenNeuronCYIntInput * d['target']})
    .attr("stroke", function (d) {return getInputEdgeStrokeColor(network, d)})
    .attr("stroke-width", function (d) {return getInputEdgeStrokeWidth(network, d)});

  for(var j = 0; j < network.layer_count - 1; j++) {
    var hiddenNeuronCYInt = (nn_svg_height - 2 * hiddenNeuronCYMin) / (network.hiddenNeurons[j + 1].neuron_count - 1 + 1e-6);

    for(var k = 0; k < network.hiddenNeurons[j].neuron_count; k++) {
      nn_svg.selectAll("g.hidden-edge" + j + "-" + k)
        .data(network.hiddenVectors[j][k])
        .enter()
        .append("g")
        .classed("hidden-edge" + j + "-" + k, true)
        .classed("edge", true)
        .append("line")
        .attr("x1", hiddenNeuronCX + j * hiddenNeuronCXInt + neuronRadius)
        .attr("x2", hiddenNeuronCX + (j + 1) * hiddenNeuronCXInt - neuronRadius)
        .attr("y1", function (d) {
          if(j == 0) {
            return hiddenNeuronCYMin + hiddenNeuronCYIntInput * d['source'];
          } else {
            hiddenNeuronCYInt = (nn_svg_height - 2 * hiddenNeuronCYMin) / (network.hiddenNeurons[j].neuron_count - 1 + 1e-6);
            
            return hiddenNeuronCYMin + hiddenNeuronCYInt * d['source'];
          }
        })
        .attr("y2", function (d) {
          hiddenNeuronCYInt = (nn_svg_height - 2 * hiddenNeuronCYMin) / (network.hiddenNeurons[j + 1].neuron_count - 1 + 1e-6);

          return hiddenNeuronCYMin + hiddenNeuronCYInt * d['target']
        })
        .attr("stroke", function (d) {return getHiddenEdgeStrokeColor(network, d)})
        .attr("stroke-width", 5);
    }
  }

  if(!network.use_hs) {
		nn_svg.selectAll("g.output-edge")
      .data(network.outputEdges)
      .enter()
      .append("g")
      .classed("output-edge", true)
      .classed("edge", true)
      .append("line")
      .attr("x1", hiddenNeuronCX + (j * hiddenNeuronCXInt) + neuronRadius)
      .attr("x2", outputNeuronCX - neuronRadius)
      .attr("y1", function (d) {return hiddenNeuronCYMin + hiddenNeuronCYIntOutput * d['source']})
      .attr("y2", function (d) {return ioNeuronCYMin + ioNeuronCYInt * d['target']})
      .attr("stroke", function (d) {return getOutputEdgeStrokeColor(network, d)})
      .attr("stroke-width", function (d) {return getOutputEdgeStrokeWidth(network, d)});    
  }

  // This function needs to be here, because it needs to "see" ioNeuronCYMin and such...
  draw_input_output_arrows = function(network) {

    if(modal_active) {
      network.inputNeurons.forEach(function(n, neuronIdx) {
        if (isCurrentContextWord(n.word)) {
          nn_svg.append("line")
            .classed("nn-io-arrow", true)  // used for erasing
            .attr("x1", "0")
            .attr("y1", ioNeuronCYMin + ioNeuronCYInt * neuronIdx)
            .attr("x2", nn_svg_width * 0.075)
            .attr("y2", ioNeuronCYMin + ioNeuronCYInt * neuronIdx)
            .attr("marker-end", "url(#marker_arrow)")
            .style("stroke", io_arrow_color())
            .style("stroke-width", "10");
        }
      });
    } else {
      network.inputNeurons.forEach(function(n, neuronIdx) {
        if (isCurrentContextWord(n.word)) {
          nn_svg.append("line")
            .classed("nn-io-arrow", true)
            .attr("x1", "0")
            .attr("y1", ioNeuronCYMin + ioNeuronCYInt * neuronIdx)
            .attr("x2", nn_svg_width * 0.075)
            .attr("y2", nn_svg_height * 0.5)
            .attr("marker-end", "url(#marker_arrow)")
            .style("stroke", io_arrow_color())
            .style("stroke-width", "10");
        }
      });
    }

    if(!network.use_hs) {
      network.outputNeurons.forEach(function(n, neuronIdx) {
        if (isCurrentTargetWord(n.word)) {
          nn_svg.append("line")
            .classed("nn-io-arrow", true)  // used for erasing
            .attr("x1", nn_svg_width)
            .attr("y1", ioNeuronCYMin + ioNeuronCYInt * neuronIdx)
            .attr("x2", nn_svg_width * (1-0.075))
            .attr("y2", ioNeuronCYMin + ioNeuronCYInt * neuronIdx)
            .attr("marker-end", "url(#marker_arrow)")
            .style("stroke", io_arrow_color())
            .style("stroke-width", "10");
        }
      });
    }

  };

  // Set up hover behavior
  d3.selectAll(".input-neuron > circle")
    .on("mouseover", mouseHoverInputNeuron)
    .on("mouseout", mouseOutInputNeuron)
    .on("click", mouseClickInputNeuron);

  d3.selectAll("#black-box")
    .on("click", setup_hidden_vis);

  $("#hidden-vis").on('hide.bs.modal', clear_hidden_vis);
}

function setup_hidden_vis() {
  modal_active = 1;
  //update neuron-vis
  setup_neural_net_svg();
  update_neural_net_svg(network_array[current_network_index]);
  setup_output(network_array[current_network_index]);
  update_output_svg(network_array[current_network_index]);
  update_neural_excite_value();

  $("#hidden-vis-clone").empty();
  $("#neuron-vis").clone().appendTo("#hidden-vis-clone");
  $("#neuron-vis").empty();
  $("#control-vis").clone().appendTo("#control-vis-clone");
}

function clear_hidden_vis() {
  $("#control-vis-clone").empty();
  $("#hidden-vis-clone").empty();
  modal_active = 0;

  //update neuron-vis
  setup_neural_net_svg();
  update_neural_net_svg(network_array[current_network_index]);
  setup_output(network_array[current_network_index]);
  update_output_svg(network_array[current_network_index]);
  update_neural_excite_value();
}

function getInputEdgeStrokeWidth(network, edge) {
  if(excite_flag) {
    return isNeuronExcited(network.inputNeurons[edge.source]) ? 5 : 0;  
  } else {
    return 5;
  }
}

function getInputEdgeStrokeColor(network, edge) {
  if(excite_flag) {
    if (isNeuronExcited(network.inputNeurons[edge.source])) return exciteValueToColor(edge.weight);
    else return "grey";  
  } else {
    return exciteValueToColor(edge.weight);
  }
}

function getHiddenEdgeStrokeColor(network, edge) {
  return exciteValueToColor(edge.weight);
}

function getOutputEdgeStrokeWidth(network, edge) {
  if(excite_flag) {
    return isNeuronExcited(network.outputNeurons[edge.target]) ? 5 : 0;  
  } else {
    return 5;
  }
}

function getOutputEdgeStrokeColor(network, edge) {
  if (excite_flag) {
    if (isNeuronExcited(network.outputNeurons[edge.target])) return exciteValueToColor(edge.weight * network.outputNeurons[edge.target].value);
    else return "grey";  
  } else {
    return exciteValueToColor(edge.weight * network.outputNeurons[edge.target].value);
  }  
}


function isNeuronExcited(neuron) {
  if (! ('value' in neuron)) {
    return false;
  } else {
    return neuron.value > 1.2 / vocabSize;  
  }
}

/*
  Only re-color some elements, without changing the neural-network structure.
*/
function update_neural_excite_value() {
  var current_network = network_array[current_network_index];

  update_internal_nodes();
  d3.selectAll("g.neuron > circle")
    .attr("fill", function(d) {return exciteValueToColor(d['value'])});
  d3.selectAll(".node--internal > circle")
  	.style("fill", function(d) {return exciteValueToColor(d['value'])});
  nn_svg.selectAll("g.input-edge > line")
    .attr("stroke-width", function(d) {return getInputEdgeStrokeWidth(current_network, d)})
    .attr("stroke", function(d) {return getInputEdgeStrokeColor(current_network, d)});
  nn_svg.selectAll("g.output-edge > line")
    .attr("stroke-width", function(d) {return getOutputEdgeStrokeWidth(current_network, d)})
    .attr("stroke", function(d) {return getOutputEdgeStrokeColor(current_network, d)});
}

// Color of arrows indicating context/target words
function io_arrow_color() {
  return "#d62728";
}

function erase_input_output_arrows() {
  nn_svg.selectAll(".nn-io-arrow").remove();
}

// Helper function
// Actual method implemented in vector_math.js
function do_feed_forward(network) {
  if(network.use_hs) {
    feedforward_with_HS(network.inputVectors, network.outputVectors, network.inputNeurons, network.hiddenNeurons, network.outputNeurons, network.outputNodes, network.hiddenVectors, network.layer_count);
  } else {
    feedforward(network.inputVectors, network.outputVectors, network.inputNeurons, network.hiddenNeurons, network.outputNeurons, network.hiddenVectors, network.layer_count);
  }
}

// Helper function
// Actual method implemented in vector_math.js
function do_backpropagate(network) {
  expectedOutput = [];
  network.outputNeurons.forEach(function(n) {
    if (isCurrentTargetWord(n.word)) {
      expectedOutput.push(1);

      if(expected_word != "") {unhighlight_path(expected_word);}
      expected_word = n.word;
      highlight_path(expected_word);

    } else {
      expectedOutput.push(0);
    }
  });

  if (network.use_hs) {
    backpropagate_with_HS(network.inputVectors, network.outputVectors, network.inputNeurons, network.hiddenNeurons, network.outputNeurons, expectedOutput, network.outputNodes, network.hiddenVectors, network.layer_count);
  } else {
    backpropagate(network.inputVectors, network.outputVectors, network.inputNeurons, network.hiddenNeurons, network.outputNeurons, expectedOutput, network.hiddenVectors, network.layer_count);  
  }
}

// Helper function
// Actual method implemented in vector_math.js
function do_apply_gradients(network) {
  if (network.use_hs) {
    apply_gradient_with_HS(network.inputVectors, network.outputVectors, network.outputNodes, getCurrentLearningRate(network), network.hidden_size, network.hiddenVectors, network.layer_count);
  } else {
    apply_gradient(network.inputVectors, network.outputVectors, getCurrentLearningRate(network), network.hiddenVectors, network.layer_count);  
  }

  //increase # of iterations
  var current_iterations = parseInt($('#iters').text());

  $('#iters').empty();
  $('#iters').append(current_iterations + 1);
}

function getCurrentLearningRate(network) {
  return network.learning_rate;
}

// Input: neural excitement level, can be positive, negative
// Output: a value between 0 and 1, for display
function exciteValueToNum(x) {
  x = x * 5;  // exaggerate it a bit
  return 1 / (1+Math.exp(-x));  // sigmoid
}

function exciteValueToColor(x) {
  return numToColor(exciteValueToNum(x));
}

function mouseHoverInputNeuron(d) {
  // Excite this neuron, inhibit others; ignore always-excited ones
  network_array.forEach(function(n,i) {
    n.inputNeurons.forEach(function(n,i) {
      if (('always_excited' in n) && n['always_excited']) {
        return;
      }
      if (i == d.idx) n['value'] = 1;
      else n['value'] = 0;
   });
  });

  // set excited flag
  excite_flag = true;

  for(var i = 0; i < network_array.length; i++) {do_feed_forward(network_array[i]);}

  update_neural_excite_value();

  // also highlight input/output vectors in heatmap
  hmap_svg
    .selectAll("g.hmap-cell")
    .selectAll('#heatin' + d.idx)
    .classed("hoverclass", true);
  hmap_svg
    .selectAll("g.hmap-cell")
    .selectAll('#heatout' + d.idx)
    .classed("hoverclass", true);
  d3.selectAll('#hmap' + d.word).attr('opacity', .5);
}

function mouseOutInputNeuron(d) {
  var excited_count = 0;

  // Inhibit all neurons, except always-excited ones
  network_array.forEach(function(n,i) {
    n.inputNeurons.forEach(function(n,i) {
      if (('always_excited' in n) && n['always_excited']) {
        return;
      }

      n['value'] = 0;
    });
  });

  // reset excited flag
  network_array[current_network_index].inputNeurons.forEach(function(n, i) {
    if (('always_excited' in n) && (n['always_excited'])) {
      ++excited_count;
    }
  });

  if(!excited_count) {
    excite_flag = false;
  }

  for(var i = 0; i < network_array.length; i++) {do_feed_forward(network_array[i]);}

  update_neural_excite_value();
 
  hmap_svg
    .selectAll("g.hmap-cell")
    .selectAll('#heatin' + d.idx)
    .classed("hoverclass", false);
  hmap_svg
    .selectAll("g.hmap-cell")
    .selectAll('#heatout' + d.idx)
    .classed("hoverclass", false);
  d3.selectAll('#hmap' + d.word).attr('opacity', 1);
}

function mouseClickInputNeuron(d) {
  if (isCurrentContextWord(d['word'])) return;

  for(var i = 0; i < network_array.length; i++) {
    var n = network_array[i].inputNeurons[d['idx']];
    
    if (('always_excited' in n) && n['always_excited']) n['always_excited'] = false;
    else n['always_excited'] = true;  
  }

  excite_flag = true;

  for(var i = 0; i < network_array.length; i++) {do_feed_forward(network_array[i]);}

  update_neural_excite_value();
}

function setup_heatmap_svg() {
  hmap_svg_width = 1000;  // view box, not physical
  hmap_svg_height = 700;  // W/H ratio should match padding-bottom in wevi.css
  d3.select('div#heatmap-vis > *').remove();
  hmap_svg = d3.select('div#heatmap-vis')
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .classed("heatmap", true)
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMinYMin meet")
   .attr("viewBox", "0 0 " + hmap_svg_width + " " + hmap_svg_height)
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .classed("heatmap-vis", true);  // for picking up svg from outside
}

function update_heatmap_svg() {
  var current_network = network_array[current_network_index];

  var inputCellBaseX = 0.15 * hmap_svg_width;
  var ioCellBaseY = 0.1 * hmap_svg_height;
  var matrixPadding = 0.05 * hmap_svg_width;
  var matrixRightMargin = 0.05 * hmap_svg_width;
  var matrixBottomMargin = 0.05 * hmap_svg_height;
  var matrixWidth = (hmap_svg_width - inputCellBaseX - matrixPadding - matrixRightMargin) / 2;
  var outputCellBaseX = inputCellBaseX + matrixWidth + matrixPadding;
  var matrixHeight = hmap_svg_height - ioCellBaseY - matrixBottomMargin;
  var cellWidth = matrixWidth / current_network.hidden_size;
  var cellHeight = matrixHeight / vocabSize;
  var cellFillWidth = 0.95 * cellWidth;
  var cellFillHeight = 0.95 * cellHeight;
  var rowLabelOffset = 0.03 * hmap_svg_width;
  var inputHeaderCX = inputCellBaseX + matrixWidth / 2;
  var outputHeaderCX = hmap_svg_width - matrixRightMargin - matrixWidth/2;
  var ioHeaderBaseY = ioCellBaseY - 0.03 * hmap_svg_height;

  var inputWeightElems = hmap_svg
    .selectAll("g.hmap-input-cell")
    .data(current_network.inputEdges)
    .enter()
    .append("g")
    .classed("hmap-input-cell", true)
    .classed("hmap-cell", true)
    .append("rect")
    .attr("x", function (d) {return inputCellBaseX + cellWidth * d['target']})
    .attr("y", function (d) {return ioCellBaseY + cellHeight * d['source']})
    .attr("width", cellFillWidth)
    .attr("height", cellFillHeight)
    .attr("id", function (d) {return 'heatin' + d['source']});

  if(!current_network.use_hs) {
  var outputWeightElems = hmap_svg
    .selectAll("g.hmap-output-cell")
    .data(current_network.outputEdges)
    .enter()
    .append("g")
    .classed("hmap-output-cell", true)
    .classed("hmap-cell", true)
    .append("rect")
    .attr("x", function (d) {return outputCellBaseX + cellWidth * d['source']})
    .attr("y", function (d) {return ioCellBaseY + cellHeight * d['target']})
    .attr("width", cellFillWidth)
    .attr("height", cellFillHeight)
    .attr("id", function(d) {return 'heatout' + d['target']});
  }

  hmap_svg
    .selectAll("g.hmap-cell > rect")
    .style("fill", function(d) {return exciteValueToColor(d['weight'])});

  hmap_svg
    .selectAll("text.hmap-vocab-label")
    .data(current_network.inputNeurons)
    .enter()
    .append("text")
    .classed("hmap-vocab-label", true)
    .text(function(d) {return d.word})
    .attr("x", inputCellBaseX - rowLabelOffset)
    .attr("y", function (d, i) {return ioCellBaseY + cellHeight * i + 0.5 * cellHeight})
    .attr("text-anchor", "end")
    .attr("alignment-baseline", "middle")
    .attr("id", function(d) {return 'hmap' + d.word})
    .style("font-size", 35);

  if(!current_network.use_hs) {
    var heatmap_labels = [
      {text: "Input Vector", x: inputHeaderCX, y: ioHeaderBaseY},
      {text: "Output Vector", x: outputHeaderCX, y: ioHeaderBaseY},
    ];    
  } else {
    var heatmap_labels = [
      {text: "Input Vector", x: inputHeaderCX, y: ioHeaderBaseY},
    ];
  }

  hmap_svg
    .selectAll("text.hmap-matrix-label")
    .data(heatmap_labels)
    .enter()
    .append("text")
    .classed("hmap-matrix-label", true)
    .attr("x", function(d){return d['x']})
    .attr("y", function(d){return d['y']})
    .text(function(d){return d['text']})
    .style("font-size", 40)
    .style("fill", "grey")
    .attr("text-anchor", "middle")
    .attr("alignment-baseline", "ideographic");
}


/*
  Updates PCA model using current input weights.
  PCA implemented in pca.js

  CURRENTLY ARCHIVED
*/

/*
function update_pca() {
  var inputWeightMatrix = []
  inputVectors.forEach(function(v) {
    var tmpRow = [];
    v.forEach(function(e) {tmpRow.push(e['weight'])});
    inputWeightMatrix.push(tmpRow);
  });
  var pca = new PCA();
  var matrixNormalized = pca.scale(inputWeightMatrix, true, true);
  principal_components = pca.pca(matrixNormalized);  // global
}

function setup_scatterplot_svg() {
  scatter_svg_width = 1000;  // view box, not physical
  scatter_svg_height = 700;  // W/H ratio should match padding-bottom in wevi.css
  d3.select('div#scatterplot-vis > *').remove();
  scatter_svg = d3.select('div#scatterplot-vis')  // global
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .classed("scatterplot", true)
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMinYMin meet")
   .attr("viewBox", "0 0 " + scatter_svg_width + " " + scatter_svg_height)
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .classed("scatterplot-vis", true);  // for picking up svg from outside
}

function update_scatterplot_svg() {
  var vectorProjections = [];
  var pc0 = principal_components[0]
  var pc1 = principal_components[1];
  inputVectors.forEach(function(v, i) {
    var tmpVec = [];
    v.forEach(function(e) {tmpVec.push(e['weight'])});
    var proj0 = dot_product(pc0, tmpVec);
    var proj1 = dot_product(pc1, tmpVec);
    vectorProjections.push({
      proj0: proj0,
      proj1: proj1,
      word: inputNeurons[i].word,
      type: 'input',
    });
  });
  outputVectors.forEach(function(v, i) {
    var tmpVec = [];
    v.forEach(function(e) {tmpVec.push(e['weight'])});
    var proj0 = dot_product(pc0, tmpVec);
    var proj1 = dot_product(pc1, tmpVec);
    vectorProjections.push({
      proj0: proj0,
      proj1: proj1,
      word: outputNeurons[i].word,
      type: 'output',
    });
  });
  hiddenNeurons.forEach(function(v, i) {
    var tmpVec = [];
    for (var j = 0; j < hiddenSize; j++) {
      tmpVec[j] = j == i ? 1 : 0;  // a column of an index matrix
    }
    var proj0 = dot_product(pc0, tmpVec);
    var proj1 = dot_product(pc1, tmpVec);
    vectorProjections.push({
      proj0: proj0,
      proj1: proj1,
      word: "h" + i,
      type: 'hidden',
    });
  });

  // Clear up SVG
  scatter_svg.selectAll("*").remove();

  // Add grid line
  var vecRenderBaseX = scatter_svg_width / 2;
  var vecRenderBaseY = scatter_svg_height / 2;
  scatter_svg.append("line")
    .classed("grid-line", true)
    .attr("x1", 0)
    .attr("x2", scatter_svg_width)
    .attr("y1", vecRenderBaseY)
    .attr("y2", vecRenderBaseY);
  scatter_svg.append("line")
    .classed("grid-line", true)
    .attr("x1", vecRenderBaseX)
    .attr("x2", vecRenderBaseX)
    .attr("y1", 0)
    .attr("y2", scatter_svg_height);
  scatter_svg.selectAll(".grid-line")
    .style("stroke", "grey")
    .style("stroke-dasharray", ("30,3"))
    .style("stroke-width", 2)
    .style("stroke-opacity", 0.75);

  var scatter_groups = scatter_svg
    .selectAll("g.scatterplot-vector")
    .data(vectorProjections)
    .enter()
    .append("g")
    .classed("scatterplot-vector", true);

  var ioVectors = scatter_groups
    .filter(function(d) {return d['type'] == "input" || d['type'] == "output"});

  ioVectors
    .append("circle")
    //.attr("x", function (d) {return d['proj0']*1000+500})
    //.attr("y", function (d) {return d['proj1']*1000+500})
    .attr("r", 10)
    .attr("stroke-width", "2")
    .attr("stroke", "grey")
    .attr("fill", getVectorColorBasedOnType);

  ioVectors
    .append("text")
    .attr("dx", "6")
    .attr("dy", "-0.25em")
    .attr("alignment-baseline", "ideographic")
    .style("font-size", 28)
    .style("fill", getVectorColorBasedOnType)
    .text(function(d) {return d.word});

  // Calculate a proper scale
  vecRenderScale = 9999999999;  // global
  vectorProjections.forEach(function(v) {
    if (v['type'] == 'input' || v['type'] == 'output') {
      vecRenderScale = Math.min(vecRenderScale, 0.4 * scatter_svg_width / Math.abs(v['proj0']));
      vecRenderScale = Math.min(vecRenderScale, 0.45 * scatter_svg_height / Math.abs(v['proj1']));
    }
  });

  ioVectors
    .attr("transform", function(d) {
      var x = d['proj0'] * vecRenderScale + vecRenderBaseX;
      var y = d['proj1'] * vecRenderScale + vecRenderBaseY;
      return "translate(" + x + ',' + y +")";
    });
}

function getVectorColorBasedOnType(d) {
  return d['type'] == "input" ? "#1f77b4" : "#ff7f0e";
}

function updatePCAButtonClick() {
  update_pca();
  update_scatterplot_svg();
}
*/


function nextButtonClick() {
  if (current_input) {
    deactivateCurrentInput();
    erase_input_output_arrows();
    
    for(var i = 0; i < network_array.length; i++) {
      do_apply_gradients(network_array[i]);  // subtract gradients from weights  
    }

    compute_perplexity();
    
    update_heatmap_svg();
    //update_scatterplot_svg();
  } else {
    activateNextInput();
    draw_input_output_arrows(network_array[current_network_index]);

    for(var i = 0; i < network_array.length; i++) {
      do_backpropagate(network_array[i]);  // compute gradients (without updating weights)
    }
  }
}

function updateAndRestartButtonClick() {
  //console.log("updated");
  $('#iters').empty();
  $('#iters').append("0");
  global_init();
}

// Train in batch
function batchTrain(numIter) {
  // Step 1:
  activateNextInput_modified();
  draw_input_output_arrows(network_array[current_network_index]);
  setTimeout(function() {

    for(var i = 0; i < network_array.length; i++) {
      do_backpropagate(network_array[i]);
    }

    // Step 2:
    deactivateCurrentInput_modified();
    erase_input_output_arrows();

    for(var i = 0; i < network_array.length; i++) {
      do_apply_gradients(network_array[i]);
    }

    compute_perplexity();

    update_heatmap_svg();
    //update_scatterplot_svg();
    if (numIter == 1) return;
    else setTimeout(function() {
      batchTrain(numIter - 1)
    }, numIter % 10 == 0 ? 50 : 0);  // when to stop for scatter plots
  }, numIter % 10 == 0 ? 50 : 0);  // when to show input/output arrows
}

/*
  For making the slides presentation only.
*/
function addColorPalette() {
  hmap_svg_width = 1000;  // view box, not physical
  hmap_svg_height = 700;  // W/H ratio should match padding-bottom in wevi.css
  d3.select('div#heatmap-vis > *').remove();
  hmap_svg = d3.select('div#heatmap-vis')
   .append("div")
   .classed("svg-container", true) //container class to make it responsive
   .classed("heatmap", true)
   .append("svg")
   //responsive SVG needs these 2 attributes and no width and height attr
   .attr("preserveAspectRatio", "xMinYMin meet")
   .attr("viewBox", "0 0 " + hmap_svg_width + " " + hmap_svg_height)
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .classed("heatmap-vis", true);  // for picking up svg from outside

  var tmpArray = [];
  for (var i = -1; i < 1; i += 0.03) {
    tmpArray.push(i);
  }

  d3.select("svg.heatmap-vis").selectAll("rect")
    .data(tmpArray)
    .enter()
    .append("rect")
    .attr("x", function (d, i) {return i * 15 + 20})
    .attr("y", 20)
    .attr("width", 17)
    .attr("height", 100)
    .style("fill", function(d) {return exciteValueToColor(d)});
}


// NEW FUNCTIONS for determining difference between HS and Original //

function vector_average(vec) {
	var tempsum = 0;

	for (var i = 0; i < vec.length; i++) {
		tempsum += vec[i];
	}

	return tempsum / vec.length;
}

function euclidean_norm(vec) {
  var tempsum = 0;

  for (var i = 0; i < vec.length; i++) {
    tempsum += Math.pow(vec[i], 2);
  }

  return Math.sqrt(tempsum);
}

function euclidean_distance(vec1, vec2) {
  assert(vec1.length == vec2.length);

  var tempsum = 0;

  for(var i = 0; i < vec1.length; i++) {
    tempsum += Math.pow((vec1[i] - vec2[i]), 2);
  }

  return Math.sqrt(tempsum);
}

function distance(vec1, vec2) {
  assert(vec1.length == vec2.length);

  var tempsum = 0;

  for(var i = 0; i < vec1.length; i++) {
    tempsum += (vec1[i] - vec2[i]);
  }

  return tempsum / vec1.length;
}

function cosine_similarity(vec1, vec2) {
  return (dot_product(vec1, vec2) / (euclidean_norm(vec1) * euclidean_norm(vec2)));
}

/*
function get_prob_vector(word, hs_option, seed) {
  // first, train model
  var modified_config_obj = {
    hidden_size: 5,
    random_state: seed,
    learning_rate: 0.2,
    use_hs: hs_option,
  };
  $('#config-text').html(JSON.stringify(modified_config_obj, null, ''));

  global_init();
  var training_sessions = 500;

  for (var i = 0; i < training_sessions; i++) {
    activateNextInput();
    do_backpropagate();
    deactivateCurrentInput();
    do_apply_gradients();
  }
  
  // Excite given neuron (for simplicity, ignore always-excited)
  var sourceindex = vocab.indexOf(word);

  inputNeurons.forEach(function(n,i) {
    if (('always_excited' in n) && n['always_excited']) {
      return;
    }
    if (i == sourceindex) n['value'] = 1;
    else n['value'] = 0;
  });
  do_feed_forward();
  update_neural_excite_value();

  // collect probabilities
  prob_vector = [];

  outputNeurons.forEach(function(n, i) {
    prob_vector.push(n['value']);
  });

  // Inhibit all neurons, except always-excited ones
  inputNeurons.forEach(function(n,i) {
    if (('always_excited' in n) && n['always_excited']) return;
    n['value'] = 0;
  });
  do_feed_forward();
  update_neural_excite_value();

  return prob_vector;
}

function get_prob_vector_distance_single(word, seed) {
  // HS training
  pvec1 = get_prob_vector(word, 1, seed);
  // Non-HS training
  pvec2 = get_prob_vector(word, 0, seed);
  
  return distance(pvec1, pvec2);
}

function get_prob_vector_distance_array(vocab, seed) {
  var tempsum = 0;

  vocab.forEach(function(n) {
    tempsum += get_prob_vector_distance_single(n, seed);
  });

  return tempsum / vocab.length;
}

function get_distance_distribution(vocab, iters) {
  distances = [];

  for (var i = 0; i < iters; i++) {
    distances.push(get_prob_vector_distance_array(vocab, (i + 1)));
  }

  return distances;
}
*/

// NEW FUNCTIONS for tree structure/visualization //

function draw_tree_interface() {

  tree_width = 450;
  tree_height = 750;

  var neuronRadius = nn_svg_width * 0.015;

  tree_svg = nn_svg.append("svg")//append("div")
   //.classed("svg-container", true) //container class to make it responsive
   //.append("svg")
   .attr("x", "50%")
   //.attr("width", "50%")
   //.attr("height", "50%")
   //.attr("float", "right")
   .attr("preserveAspectRatio", "xMinYMin meet")
   .attr("viewBox", "0 -25 " + tree_width + " " + tree_height)
   //class to make it responsive
   .classed("svg-content-responsive", true)
   .classed("tree", true);  // for picking up svg from outside

  g = tree_svg.append("g").attr("transform", "translate(120,0)");

  var tree = d3.cluster()
    .size([tree_height - 50, tree_width - 150]);

  root = d3.hierarchy(JSON.parse(JSONtree));
  //.sort(function(a, b) { return (a.height - b.height) || a.name.localeCompare(b.name); });

  tree(root);

  var link = g.selectAll(".link")
    .data(root.descendants().slice(1))
    .enter().append("path")
    .attr("class", "link")
      .attr("d", function(d) {
        return "M" + d.y + "," + d.x
            + "C" + (d.parent.y + 10) + "," + d.x
            + " " + (d.parent.y + 10) + "," + d.parent.x
            + " " + d.parent.y + "," + d.parent.x;
      });

  var node = g.selectAll(".node")
    .data(root.descendants())
    .enter().append("g")
    .attr("class", function(d) { return "node" + (d.children ? " node--internal" : " node--leaf"); })
    .attr("transform", function(d) { return "translate(" + d.y + "," + d.x + ")"; })

  var internals = g.selectAll(".node--internal")
    .append("circle")
    .attr("r", neuronRadius)
    .attr("fill", "lightgrey");

  node.append("text")
    .attr("dy", 3)
    .attr("x", function(d) { return d.children ? -8 : 25; })
    .style("font-size", 28)
    .style("text-anchor", function(d) { return d.children ? "end" : "start"; })
    .text(function(d) {return d.data.name;});

  var leaves = g.selectAll(".node--leaf")
    .attr("word", function(d) {return d.data.name;})
    .attr("class", function(d) {return d3.select(this).attr("class") + " " + d.data.name;});

  d3.selectAll(".node--internal").each(init_node_vect);
  generate_words();
}

function set_words(node) {
	if(node.children) {
		node.words = [];
	}
}

function populate_words(node) {
	if(node.children) {
		if(node.children[0].data.name != "") {
			node.words.push(node.children[0].data.name);
		}
		if(node.children[1].data.name != "") {
			node.words.push(node.children[1].data.name);
		}

		if(node.children[0].words){
			node.words = node.words.concat(node.children[0].words);
		}
		if(node.children[1].words){
			node.words = node.words.concat(node.children[1].words);
		}			
	}
}

function make_words_unique(node) {
	if(node.words) {
		node.words = node.words.filter((v, i, a) => a.indexOf(v) === i);
	}
}

function populate_word_class(node) {
	if(node.words) {
		var path_words = node.words.join(" ");

		d3.select(this).attr("class", d3.select(this).attr("class") + " " + path_words);
	}
}

function generate_words() {
	tree_svg.selectAll(".node").each(set_words);

	for (var i = 0; i < root.height; i++) {
		tree_svg.selectAll(".node").each(populate_words);
	}

	tree_svg.selectAll(".node").each(make_words_unique);

	tree_svg.selectAll(".node").each(populate_word_class);
}

function highlight_path(string) {
	d3.selectAll(".node--internal." + string)
		.select("circle")
		.style("fill", "green");
}

function unhighlight_path(string) {
	d3.selectAll(".node--internal." + string)
		.select("circle")
		.style("fill", "lightgrey");

	update_neural_excite_value();
}

// first element in children array is LEFT child (according to d3 hierarchy)

function update_internal_children(node, tree_root) {
	var tempnode = node;
	var temproot = tree_root;

	tempnode.vect = tree_root.vect;
	tempnode.value = dot_product(tempnode.vect, hiddenVector) / 5; //make it a little less dramatic

	if(tempnode.children) {
		var childname0 = tempnode.children[0].data.name;
		var childname1 = tempnode.children[1].data.name;

		if(childname0 == "") {
			update_internal_children(tempnode.children[0], temproot.left);
		}
		
		if(childname1 == "") {
			update_internal_children(tempnode.children[1], temproot.right);
		}
	}
}

function update_internal_helper(node) {
	var tempnode = this.__data__;
	var temproot = network_array[current_network_index].hufftree.root;

	tempnode.vect = temproot.vect;
	tempnode.value = dot_product(tempnode.vect, hiddenVector) / 5; //make it a little less dramatic

	if(tempnode.children) {
		var childname0 = tempnode.children[0].data.name;
		var childname1 = tempnode.children[1].data.name;

		if(childname0 == "") {
			update_internal_children(tempnode.children[0], temproot.left);
		}
		
		if(childname1 == "") {
			update_internal_children(tempnode.children[1], temproot.right);
		}
	}
}

function update_internal_nodes() {
  var current_network = network_array[current_network_index];
  hiddenVector = [];

  for(var i = 0; i < current_network.hiddenNeurons[current_network.layer_count - 1].neuron_count; i++) {
    hiddenVector.push(current_network.hiddenNeurons[current_network.layer_count - 1].data[i].value);
  }

	d3.select(".node--internal").each(update_internal_helper);
}

function init_node_vect(node) {
  hiddenVector = [];

	node.vect = [];
	node.value = 0;

	for (var i = 0; i < network_array[current_network_index].hidden_size; i++) {
		node.vect[i] = 0;
    hiddenVector[i] = 0;
	}
}

function setup_output(network) {
  output_svg_width = 200;  // view box, not physical
  output_svg_height = 400;  // W/H ratio should match padding-bottom in wevi.css
  d3.select('div#output-vis' + network.id + ' > *').remove();
  output_svg = d3.select('div#output-vis' + network.id)
   .append("div")
   //.classed("svg-container", true) //container class to make it responsive
   //.classed("output", true)
   .append("svg")
   .attr("width", output_svg_width)
   .attr("height", output_svg_height);
   //responsive SVG needs these 2 attributes and no width and height attr
   //.attr("preserveAspectRatio", "xMinYMin meet")
   //.attr("viewBox", "0 0 " + output_svg_width + " " + output_svg_height)
   //class to make it responsive
   //.classed("svg-content-responsive", true)
   //.classed("output", true);  // for picking up svg from outside

  var inputNeuronCX = output_svg_width * 0.44;
  var ioNeuronCYMin = output_svg_height * 0.10;
  var ioNeuronCYInt = (output_svg_height - 6.5 * ioNeuronCYMin) / (vocabSize - 1 + 1e-6);
  var neuronRadius = output_svg_width * 0.045;
  var neuronLabelOffset = neuronRadius * 2;

  d3.select('#output-vis0 > *').remove();
  general_output_svg = d3.select('#output-vis0')
    .append("div")
    .append("svg")
    .attr("width", output_svg_width)
    .attr("height", output_svg_height);

  var general_tabOutputNeurons = general_output_svg
    .selectAll("g.output-neuron")
    //.data(network.outputNeuronElems.data().slice(0, vocabSize))
    .data(network.outputNeurons)
    .enter()
    .append("g")
    .classed("output-neuron", true)
    .classed("neuron", true);

 general_tabOutputNeurons
  .append("text")
  .classed("neuron-label", true)
  .attr("x", inputNeuronCX * .8)
  .attr("y", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * 2 * i})
  .attr("text-anchor", "end");

  general_output_svg.selectAll(".neuron-label")
    .attr("alignment-baseline", "middle")
    .style("font-size", 16)
    .text(function(d) {return d.word});
}

function update_output_svg(network) { 
  var inputNeuronCX = output_svg_width * 0.25;
  var ioNeuronCYMin = output_svg_height * 0.10;
  var ioNeuronCYInt = (output_svg_height - 6.5 * ioNeuronCYMin) / (vocabSize - 1 + 1e-6);
  var neuronRadius = output_svg_width * 0.045;
  var neuronLabelOffset = neuronRadius * 2;

  var tabOutputNeurons = output_svg
    .selectAll("g.output-neuron")
    //.data(network.outputNeuronElems.data().slice(0, vocabSize))
    .data(network.outputNeurons)
    .enter()
    .append("g")
    .classed("output-neuron", true)
    .classed("neuron", true);

  tabOutputNeurons
    .append("circle")
    .attr("cx", inputNeuronCX)
    .attr("cy", function(d, i) {return ioNeuronCYMin + ioNeuronCYInt * 2 * i});

  output_svg.selectAll("g.neuron > circle")
    .attr("r", neuronRadius)
    .attr("stroke-width", "2")
    .attr("stroke", "grey")
    .attr("fill", function(d) {return numToColor(0.5);});

/*
 tabOutputNeurons
    .append("text")
    .classed("neuron-label", true)
    .attr("x", inputNeuronCX - neuronLabelOffset)
    .attr("y", function (d, i) {return ioNeuronCYMin + ioNeuronCYInt * 2 * i})
    .attr("text-anchor", "end");

  output_svg.selectAll(".neuron-label")
    .attr("alignment-baseline", "middle")
    .style("font-size", 16)
    .text(function(d) {return d.word});
*/
}

function compute_perplexity() {

  var current_neuron_idx = 0;

  temp_prob = [];

  for(var i = 0; i < network_array.length; i++) {
    temp_prob.push(1);
  }

  for(var current_neuron_idx = 0; current_neuron_idx < vocabSize - 1; current_neuron_idx++) {
    
    // Excite neurons to get proper probabilities
    network_array.forEach(function(n,i) {
      n.inputNeurons.forEach(function(n,i) {
        if (('always_excited' in n) && n['always_excited']) {
          return;
        }
        if (i == current_neuron_idx) n['value'] = 1;
        else n['value'] = 0;
     });
    });

    for(var i = 0; i < network_array.length; i++) {do_feed_forward(network_array[i]);}

    // Grab probability
    for(var i = 0; i < network_array.length; i++) {
      temp_prob[i] *= network_array[i].outputNeurons[current_neuron_idx + 1]['value'];
    } 

  }

  // Inhibit neurons
  network_array.forEach(function(n,i) {
    n.inputNeurons.forEach(function(n,i) {
      if (('always_excited' in n) && n['always_excited']) return;
      n['value'] = 0;
    });
  });

  for(var i = 0; i < network_array.length; i++) {do_feed_forward(network_array[i]);}

  update_neural_excite_value();

  // Update perplexity
  for(var i = 0; i < network_array.length; i++) {
    network_array[i].perplexity.push(Math.pow(temp_prob[i], (-1 * Math.pow(vocabSize, -1))));
  }

}