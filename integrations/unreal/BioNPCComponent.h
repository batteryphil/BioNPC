// Copyright (c) BioNPC Contributors. MIT License.
// Ready-to-drop-in Unreal Engine 5 ActorComponent & MassEntity wrapper for BioNPC.
// Links directly against `libbionpc.a` / `bionpc.h`.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "bionpc.h"
#include "BioNPCComponent.generated.h"

/**
 * UBioNPCComponent
 *
 * High-performance biological CTRNN Hindbrain component for Unreal Engine 5 actors.
 * Runs 60+ FPS continuous sensorimotor reflexes with Drosophila APL neurotransmitter decay,
 * while exposing full Designer Controllability (clamps, state locks, and Forebrain spikes)
 * to Blueprints and Behavior Tree decorators.
 */
UCLASS(ClassGroup=(AI), meta=(BlueprintSpawnableComponent))
class UBioNPCComponent : public UActorComponent
{
    GENERATED_BODY()

public:
    UBioNPCComponent()
    {
        PrimaryComponentTick.bCanEverTick = true;
        bionpc_control_init(&Control);
        FMemory::Memzero(&AgentState, sizeof(bionpc_agent_state_t));
    }

    /** Path to exported .bionpc binary weights relative to Content/BioNPC/ */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BioNPC|Archetype")
    FString WeightsFile = TEXT("archetypes/apex_predator.bionpc");

    /** Optional path to .ini Designer Tuning Profile */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BioNPC|Archetype")
    FString DesignerProfileIni = TEXT("archetypes/apex_predator.ini");

    /** Designer sensory multipliers (e.g. [PreySight, ThreatSight, Hunger, Aggression]) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BioNPC|Designer Control")
    TArray<float> SensoryScale = { 1.0f, 1.0f, 1.0f, 1.0f };

    /** Designer sensory biases (e.g. temporary Forebrain emotion offsets) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BioNPC|Designer Control")
    TArray<float> SensoryBias = { 0.0f, 0.0f, 0.0f, 0.0f };

    /** Lock biological chemical decay (e.g. during cutscenes or scripted boss phases) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "BioNPC|Designer Control")
    bool bLockChemicalState = false;

    /** Latest continuous motor outputs computed on Tick */
    UPROPERTY(BlueprintReadOnly, Category = "BioNPC|Outputs")
    TArray<float> MotorActions;

    virtual void BeginPlay() override
    {
        Super::BeginPlay();
        FString FullWeightsPath = FPaths::ProjectContentDir() / WeightsFile;
        Model = bionpc_model_load_file(TCHAR_TO_UTF8(*FullWeightsPath));

        if (!DesignerProfileIni.IsEmpty())
        {
            FString FullIniPath = FPaths::ProjectContentDir() / DesignerProfileIni;
            bionpc_load_ini_profile(TCHAR_TO_UTF8(*FullIniPath), Model, &Control);
        }

        if (Model)
        {
            MotorActions.SetNumZeroed(Model->config.d_out);
        }
    }

    virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override
    {
        if (Model)
        {
            bionpc_model_free(Model);
            Model = nullptr;
        }
        Super::EndPlay(EndPlayReason);
    }

    /**
     * Execute a single 60 FPS Hindbrain tick from continuous sensory floats.
     */
    UFUNCTION(BlueprintCallable, Category = "BioNPC|Inference")
    TArray<float> StepBrain(const TArray<float>& Senses)
    {
        if (!Model || Senses.Num() < Model->config.d_in)
        {
            return MotorActions;
        }

        Control.apl_locked = bLockChemicalState ? 1 : 0;
        for (int32 i = 0; i < FMath::Min(Senses.Num(), BIONPC_MAX_DIM); ++i)
        {
            if (i < SensoryScale.Num()) Control.sensory_scale[i] = SensoryScale[i];
            if (i < SensoryBias.Num())  Control.sensory_bias[i]  = SensoryBias[i];
        }

        MotorActions.SetNumUninitialized(Model->config.d_out);
        bionpc_forward_single_controlled(
            Model,
            Senses.GetData(),
            MotorActions.GetData(),
            &AgentState,
            &Control
        );
        return MotorActions;
    }

    /**
     * Inject an instantaneous chemical/neurotransmitter spike from the LLM Forebrain
     * or a Blueprint gameplay event (e.g. player threatens NPC or offers food).
     */
    UFUNCTION(BlueprintCallable, Category = "BioNPC|Forebrain")
    void InjectChemicalSpike(float DeltaAPL, int32 LayerIndex = -1)
    {
        bionpc_state_spike(&AgentState, LayerIndex, DeltaAPL);
    }

    /**
     * Override or clamp a specific motor action channel (e.g. disable Attack in Sanctuary zones).
     */
    UFUNCTION(BlueprintCallable, Category = "BioNPC|Designer Control")
    void SetActionOverride(int32 ActionIndex, bool bEnableOverride, float ForcedValue = -10.0f)
    {
        if (ActionIndex >= 0 && ActionIndex < BIONPC_MAX_DIM)
        {
            Control.action_override_mask[ActionIndex] = bEnableOverride ? 1 : 0;
            Control.action_override_val[ActionIndex]  = ForcedValue;
        }
    }

    /**
     * Compute deterministic IEEE-754 CRC32 state checksum for multiplayer replication verification.
     */
    UFUNCTION(BlueprintCallable, Category = "BioNPC|Multiplayer")
    int32 GetStateChecksum() const
    {
        if (!Model) return 0;
        return static_cast<int32>(bionpc_crc32_states(&AgentState, 1, Model->config.n_layers));
    }

private:
    bionpc_model_t* Model = nullptr;
    bionpc_agent_state_t AgentState;
    bionpc_control_t Control;
};
