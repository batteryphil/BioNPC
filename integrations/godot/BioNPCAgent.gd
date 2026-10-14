extends CharacterBody3D
class_name BioNPCAgent

## BioNPC Godot 4 Agent Component with Live 3D Chemical Debug HUD
## Supports 60 FPS biological CTRNN reflexes, Designer Control overrides,
## and Hybrid LLM Forebrain chemical spikes.

@export_enum("apex_predator", "herd_prey", "territorial_guard", "cowardly_scavenger", "swarm_minion")
var archetype: String = "apex_predator"

@export_group("Biological Chemistry")
@export_range(0.0, 1.0) var hunger: float = 0.25
@export_range(0.0, 1.0) var aggression: float = 0.40
@export_range(0.0, 1.0) var fear: float = 0.10
@export var apl_decay: float = 0.88
@export var apl_gain: float = 0.12
@export var lock_chemical_state: bool = false

@export_group("Designer Overrides")
@export var disable_attack_in_safe_zone: bool = false
@export var show_debug_hud: bool = true

var apl_state: float = 0.0
var last_actions: Array[float] = [0.0, 0.0, 0.0]
var debug_label: Label3D = null

func _ready() -> void:
	if show_debug_hud:
		_create_debug_hud()

func _create_debug_hud() -> void:
	debug_label = Label3D.new()
	debug_label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
	debug_label.position = Vector3(0.0, 2.2, 0.0)
	debug_label.font_size = 24
	debug_label.outline_size = 6
	add_child(debug_label)
	_update_debug_hud()

func inject_forebrain_spike(delta_aggression: float, delta_fear: float, delta_apl: float) -> void:
	if lock_chemical_state:
		return
	aggression = clampf(aggression + delta_aggression, 0.0, 1.0)
	fear = clampf(fear + delta_fear, 0.0, 1.0)
	apl_state = clampf(apl_state + delta_apl, -5.0, 5.0)
	_update_debug_hud()

func step_hindbrain(prey_sight: float, threat_sight: float, delta: float) -> Array[float]:
	if not lock_chemical_state:
		hunger = clampf(hunger + 0.01 * delta, 0.0, 1.0)
		var activity: float = (absf(prey_sight) + absf(threat_sight) + hunger + aggression) * 0.25
		apl_state = apl_state * apl_decay + activity * apl_gain
		fear = lerpf(fear, threat_sight * (1.0 - aggression * 0.5), 0.15)

	var inhibition: float = 1.0 - apl_state
	var chase: float = clampf((prey_sight * 1.4 + hunger * 0.8 - fear * 1.2) * inhibition, 0.0, 1.0)
	var attack: float = clampf((threat_sight * 0.8 + aggression * 1.5 - fear * 0.9) * inhibition, 0.0, 1.0)
	var flee: float = clampf((threat_sight * 1.5 + fear * 1.2 - aggression * 0.8) * inhibition, 0.0, 1.0)

	if disable_attack_in_safe_zone:
		attack = 0.0

	last_actions = [chase, attack, flee]
	if show_debug_hud:
		_update_debug_hud()
	return last_actions

func _bar(val: float) -> String:
	var filled: int = int(clampf(val, 0.0, 1.0) * 8.0)
	return "█".repeat(filled) + "░".repeat(8 - filled)

func _update_debug_hud() -> void:
	if debug_label == null:
		return
	var state_tag := "[LOCKED]" if lock_chemical_state else ""
	debug_label.text = (
		"%s %s\n" % [archetype.to_upper(), state_tag] +
		"FEAR  [%s] %.2f\n" % [_bar(fear), fear] +
		"HUNGR [%s] %.2f\n" % [_bar(hunger), hunger] +
		"AGGRO [%s] %.2f\n" % [_bar(aggression), aggression] +
		"APL: %+.3f | Act: C=%.2f A=%.2f F=%.2f" % [apl_state, last_actions[0], last_actions[1], last_actions[2]]
	)
