// Copyright 2026 The Binius Developers

//! The verifier for the M4 constraint reduction: the AND-check followed by the shift reduction.
//!
//! This mirrors the prover's composition on the same transcript:
//!
//! 1. The AND-check verifies `A & B == C` over all rows, yielding operand claims at a row point.
//! 2. That point's low coordinates are the instance index `r_rho`, its high coordinates `r_x`.
//! 3. The shift reduction reduces the operand claims to one evaluation of the folded witness.
//! 4. The public-input consistency check ties in the shared constants.
//!
//! When the circuit has MUL constraints the IntMul check verifies too.
//! It yields per-bit operand claims at its own instance point, distinct from the AND-check's.
//! A batched multilinear-evaluation sumcheck unifies both onto one shared `r_rho`.
//! Both operand claims at that point then feed the shift.
//!
//! The output is that witness claim together with `r_rho`.
//! The caller binds it to the committed trace by ring-switching at `r_j || r_rho || r_y`.
//! Evaluating the trace's instance coordinates at `r_rho` performs that instance fold.

use binius_core::{constraint_system::ConstraintSystem, word::Word};
use binius_field::{AESTowerField8b as B8, Field};
use binius_iop::channel::IOPVerifierChannel;
use binius_ip::sumcheck::{BatchSumcheckOutput, batch_verify};
use binius_math::{
	BinarySubspace,
	inner_product::inner_product,
	multilinear::eq::eq_ind,
	univariate::{evaluate_univariate, lagrange_evals_scalars},
};
use binius_utils::checked_arithmetics::checked_log_2;
use binius_verifier::{
	Error,
	config::B128,
	protocols::{
		bitand::AndCheckOutput,
		intmul::{IntMulOutput, verify as verify_intmul_reduction},
		shift::{self, BITAND_ARITY, INTMUL_ARITY, OperatorData, VerifyOutput},
	},
};

use crate::verify_bitand_reduction;

/// The verifier's output of the M4 constraint reduction.
pub struct ReductionVerifierOutput {
	/// The shared instance-fold challenge the witness is folded at.
	///
	/// Without MUL constraints this is the low half of the AND-check row point.
	/// With MUL constraints the re-randomization sumcheck produces it fresh.
	pub r_rho: Vec<B128>,
	/// The reduced claim about the instance-folded witness, from the shift reduction.
	pub shift: VerifyOutput<B128>,
}

/// Verifies the AND-check and shift reduction over the batch on one transcript.
///
/// # Arguments
///
/// - `cs`: the prepared single-instance constraint system shared by every instance.
/// - `log_instances`: base-2 logarithm of the instance count.
/// - `channel`: the verifier channel reading messages and redrawing Fiat-Shamir challenges.
///
/// # Errors
///
/// Returns an error if the AND-check, the shift reduction, or the public-input check fails.
pub fn verify_reduction<Channel>(
	cs: &ConstraintSystem,
	log_instances: usize,
	channel: &mut Channel,
) -> Result<ReductionVerifierOutput, Error>
where
	Channel: IOPVerifierChannel<B128, Elem = B128>,
{
	// One base domain shared by the AND-check, the shift, and the IntMul operand collapse.
	// The AND-check's univariate-skip domain spans one dimension above the 64-bit word.
	let andcheck_domain = BinarySubspace::<B8>::with_dim(Word::LOG_BITS + 1);
	// The shift domain drops that extra dimension.
	let shift_domain = andcheck_domain
		.reduce_dim(Word::LOG_BITS)
		.isomorphic::<B128>();

	// SOUNDNESS: the IntMul check verifies before the BitAnd check, mirroring the prover.
	// Its per-bit operand evaluations are read from the transcript here.
	// BitAnd then draws the univariate challenge that collapses them.
	// Reading them first stops a malicious prover choosing them as a function of that challenge.
	// Do not reorder these.
	//
	// The IntMul columns span every instance's constraints.
	// So the check runs over `log_instances + log_n_mul` row variables.
	let intmul_output = if cs.n_mul_constraints() > 0 {
		let log_n_mul = checked_log_2(cs.n_mul_constraints());
		Some(verify_intmul_reduction::<B128, _>(log_instances + log_n_mul, channel)?)
	} else {
		None
	};

	// AND-check over all `K * n_and` rows.
	let log_n_and = checked_log_2(cs.and_constraints.len());
	let AndCheckOutput {
		a_eval,
		b_eval,
		c_eval,
		z_challenge,
		eval_point,
	} = verify_bitand_reduction(log_instances + log_n_and, &andcheck_domain, channel)?;

	// The AND-check row point is `r_rho_and || r_x_and`: the instance index low, the constraint
	// index high.
	let (r_rho_and, r_x_and) = eval_point.split_at(log_instances);

	// Reduce to one shared instance point and both operand claims at it.
	let (r_rho, bitand, intmul) = match intmul_output {
		Some(intmul_output) => {
			// Both operations enter the re-randomization as operand claims at their own instance
			// point. BitAnd is already oblong; IntMul is collapsed from its per-bit form.
			let lagrange = lagrange_evals_scalars::<B128, B128>(&shift_domain, z_challenge);
			RerandomizedOperations {
				bitand: OperationClaim::new([a_eval, b_eval, c_eval], r_x_and, r_rho_and),
				intmul: OperationClaim::from_intmul(intmul_output, &lagrange, log_instances),
			}
			.verify(channel)?
		}
		// No MUL constraints: the AND-check instance point is used directly.
		// The IntMul claim is a zero claim at an empty point.
		None => (
			r_rho_and.to_vec(),
			OperatorData::new(r_x_and.to_vec(), [a_eval, b_eval, c_eval]),
			OperatorData::new(Vec::new(), [B128::ZERO; 4]),
		),
	};

	// Reduce the operand claims to one witness evaluation.
	let shift = shift::verify::<B128, _>(cs, &bitand, &intmul, channel)?;

	// Tie in the shared constants through the public-input consistency check.
	// The shift evaluates them over the layout's power-of-two word count.
	// Their count need not be a power of two, so they are passed unpadded.
	shift::check_eval::<B128, _>(
		cs,
		&cs.constants,
		&bitand,
		&intmul,
		&shift_domain,
		z_challenge,
		&shift,
		channel,
	)?;

	Ok(ReductionVerifierOutput { r_rho, shift })
}

/// The degree of the re-randomization's round polynomials.
///
/// Each operand is a multilinear evaluation, expressed with the quadratic evaluator.
/// Its degree-2 prime polynomial gains one degree from the equality factor, giving 3.
// TODO: a degree-1 multilinear-eval store evaluator would drop this to 2; none exists yet.
const RERAND_DEGREE: usize = 3;

/// The shared instance point together with both operations' operand data at that point.
type RerandOutput = (Vec<B128>, OperatorData<B128, BITAND_ARITY>, OperatorData<B128, INTMUL_ARITY>);

/// One operation's oblong operand claims and the points they are claimed at.
///
/// The AND-check and the IntMul check both reduce to this shape.
/// The re-randomization transports the claims to the instance point shared by both operations.
struct OperationClaim<const ARITY: usize> {
	/// The oblong operand claim per operand, in operand order.
	operand_claims: [B128; ARITY],
	/// The constraint-index point the operands are claimed at.
	r_x: Vec<B128>,
	/// The instance-index point the operands are claimed at.
	r_rho: Vec<B128>,
}

impl<const ARITY: usize> OperationClaim<ARITY> {
	/// The operand claims at the constraint point `r_x` and instance point `r_rho`.
	fn new(operand_claims: [B128; ARITY], r_x: &[B128], r_rho: &[B128]) -> Self {
		Self {
			operand_claims,
			r_x: r_x.to_vec(),
			r_rho: r_rho.to_vec(),
		}
	}
}

impl OperationClaim<INTMUL_ARITY> {
	/// Builds the IntMul claim by collapsing its per-bit operand claims to oblong claims.
	///
	/// The Lagrange weights fold the per-bit claims at the univariate challenge.
	/// This gives the oblong form the BitAnd claims already have.
	/// The IntMul row point splits into an instance part (low) and a constraint part (high).
	fn from_intmul(
		intmul_output: IntMulOutput<B128>,
		lagrange: &[B128],
		log_instances: usize,
	) -> Self {
		let IntMulOutput {
			eval_point: r_out_mul,
			a_evals,
			b_evals,
			c_lo_evals,
			c_hi_evals,
		} = intmul_output;
		let oblong = |evals: Vec<B128>| inner_product(evals, lagrange.iter().copied());
		let (r_rho, r_x) = r_out_mul.split_at(log_instances);
		Self::new(
			[
				oblong(a_evals),
				oblong(b_evals),
				oblong(c_lo_evals),
				oblong(c_hi_evals),
			],
			r_x,
			r_rho,
		)
	}
}

/// The two operations' claims entering the batched instance re-randomization.
struct RerandomizedOperations {
	/// The BitAnd operand claims at the AND-check instance point.
	bitand: OperationClaim<BITAND_ARITY>,
	/// The IntMul operand claims at the IntMul instance point.
	intmul: OperationClaim<INTMUL_ARITY>,
}

impl RerandomizedOperations {
	/// Verifies the batched sumcheck that unifies the two operations' instance points.
	///
	/// - Check the sumcheck transporting every operand claim onto one shared instance point.
	/// - Read the reduced operand evaluations at that point.
	/// - Bind them to the sumcheck.
	///
	/// # Returns
	///
	/// The shared instance point, the BitAnd operand data, and the IntMul operand data.
	fn verify<Channel>(self, channel: &mut Channel) -> Result<RerandOutput, Error>
	where
		Channel: IOPVerifierChannel<B128, Elem = B128>,
	{
		// Both operations reduce over the same instance axis; recover its width from either point.
		let log_instances = self.bitand.r_rho.len();

		// Verify the batched sumcheck: one multilinear-eval claim per operand, ordered
		// [BitAnd a, b, c | IntMul a, b, lo, hi].
		let sums: Vec<B128> = self
			.bitand
			.operand_claims
			.iter()
			.copied()
			.chain(self.intmul.operand_claims)
			.collect();
		let BatchSumcheckOutput {
			batch_coeff,
			eval,
			mut challenges,
		} = batch_verify(log_instances, RERAND_DEGREE, &sums, channel)?;
		challenges.reverse();
		let r_rho = challenges;

		// The prover wrote the reduced operand evaluations at `r_rho`, grouped by operation.
		// These are the operand claims the shift consumes.
		let bitand_evals = channel.recv_array::<BITAND_ARITY>()?;
		let intmul_evals = channel.recv_array::<INTMUL_ARITY>()?;

		// Bind the reduced evals to the sumcheck: each claim's contribution is its
		// eq(instance_point, r_rho) weight times its reduced eval, batched by `batch_coeff`.
		let eq_and = eq_ind(&self.bitand.r_rho, &r_rho);
		let eq_mul = eq_ind(&self.intmul.r_rho, &r_rho);
		let expected: Vec<B128> = bitand_evals
			.map(|eval| eq_and * eval)
			.into_iter()
			.chain(intmul_evals.map(|eval| eq_mul * eval))
			.collect();
		channel.assert_zero(evaluate_univariate(&expected, batch_coeff) - eval)?;

		let bitand_data = OperatorData::new(self.bitand.r_x, bitand_evals);
		let intmul_data = OperatorData::new(self.intmul.r_x, intmul_evals);
		Ok((r_rho, bitand_data, intmul_data))
	}
}
