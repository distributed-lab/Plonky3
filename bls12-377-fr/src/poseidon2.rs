//! Diffusion matrix for Bls12-377
//!
//! Even tho the reference is for the other field, we used it for BLS12-377Fr considering the common
//! field nature.
//! Reference: https://github.com/HorizenLabs/poseidon2/blob/main/plain_implementations/src/poseidon2/poseidon2_instance_bls12.rs

use std::sync::OnceLock;

use p3_field::FieldAlgebra;
use p3_poseidon2::{
    add_rc_and_sbox_generic, external_initial_permute_state, external_terminal_permute_state,
    internal_permute_state, matmul_internal, ExternalLayer, ExternalLayerConstants,
    ExternalLayerConstructor, HLMDSMat4, InternalLayer, InternalLayerConstructor, Poseidon2,
};

use crate::Bls12_377Fr;

/// Degree of the chosen permutation polynomial for BLS12-377, used as the Poseidon2 S-Box.
///
/// As p - 1 is divisible by 2 and 3 the smallest choice for a degree D satisfying gcd(p - 1, D) = 1 is 5.
const BLS12_337_S_BOX_DEGREE: u64 = 5;

/// An implementation of the Poseidon2 hash function for the [Bls12_377Fr] field.
///
/// It acts on arrays of the form `[Bls12_377Fr; WIDTH]`.
pub type Poseidon2Bls12337<const WIDTH: usize> = Poseidon2<
    Bls12_377Fr,
    Poseidon2ExternalLayerBls12337<WIDTH>,
    Poseidon2InternalLayerBls12337,
    WIDTH,
    BLS12_337_S_BOX_DEGREE,
>;

/// Currently we only support a single width for Poseidon2 Bls12377Fr.
const BLS12_377_WIDTH: usize = 3;

#[inline]
fn get_diffusion_matrix_3() -> &'static [Bls12_377Fr; 3] {
    static MAT_DIAG3_M_1: OnceLock<[Bls12_377Fr; 3]> = OnceLock::new();
    MAT_DIAG3_M_1.get_or_init(|| [Bls12_377Fr::ONE, Bls12_377Fr::ONE, Bls12_377Fr::TWO])
}

#[derive(Debug, Clone, Default)]
pub struct Poseidon2InternalLayerBls12337 {
    internal_constants: Vec<Bls12_377Fr>,
}

impl InternalLayerConstructor<Bls12_377Fr> for Poseidon2InternalLayerBls12337 {
    fn new_from_constants(internal_constants: Vec<Bls12_377Fr>) -> Self {
        Self { internal_constants }
    }
}

impl InternalLayer<Bls12_377Fr, BLS12_377_WIDTH, BLS12_337_S_BOX_DEGREE>
    for Poseidon2InternalLayerBls12337
{
    /// Perform the internal layers of the Poseidon2 permutation on the given state.
    fn permute_state(&self, state: &mut [Bls12_377Fr; BLS12_377_WIDTH]) {
        internal_permute_state::<Bls12_377Fr, BLS12_377_WIDTH, BLS12_337_S_BOX_DEGREE>(
            state,
            |x| matmul_internal(x, *get_diffusion_matrix_3()),
            &self.internal_constants,
        )
    }
}

pub type Poseidon2ExternalLayerBls12337<const WIDTH: usize> =
    ExternalLayerConstants<Bls12_377Fr, WIDTH>;

impl<const WIDTH: usize> ExternalLayerConstructor<Bls12_377Fr, WIDTH>
    for Poseidon2ExternalLayerBls12337<WIDTH>
{
    fn new_from_constants(external_constants: ExternalLayerConstants<Bls12_377Fr, WIDTH>) -> Self {
        external_constants
    }
}

impl<const WIDTH: usize> ExternalLayer<Bls12_377Fr, WIDTH, BLS12_337_S_BOX_DEGREE>
    for Poseidon2ExternalLayerBls12337<WIDTH>
{
    /// Perform the initial external layers of the Poseidon2 permutation on the given state.
    fn permute_state_initial(&self, state: &mut [Bls12_377Fr; WIDTH]) {
        external_initial_permute_state(
            state,
            self.get_initial_constants(),
            add_rc_and_sbox_generic::<_, BLS12_337_S_BOX_DEGREE>,
            &HLMDSMat4,
        );
    }

    /// Perform the terminal external layers of the Poseidon2 permutation on the given state.
    fn permute_state_terminal(&self, state: &mut [Bls12_377Fr; WIDTH]) {
        external_terminal_permute_state(
            state,
            self.get_terminal_constants(),
            add_rc_and_sbox_generic::<_, BLS12_337_S_BOX_DEGREE>,
            &HLMDSMat4,
        );
    }
}
