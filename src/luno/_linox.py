import jax
from jax import numpy as jnp
import linox
from linox._arithmetic import CongruenceTransform, ScaledLinearOperator


@linox.diagonal.dispatch
def _(A: linox.SymmetricLowRank) -> jax.Array:
    return jnp.sum(A.U**2 * A.S, axis=-1)


@linox.diagonal.dispatch
def _(A: ScaledLinearOperator) -> jax.Array:
    return A.scalar * linox.diagonal(A.operator)


########################################################################################
# Congruence Transforms ################################################################
########################################################################################


##########################################################
# (LinearOperator, IsotropicScalingPlusSymmetricLowRank) #
##########################################################


class CongruenceTransform_LinearOperator_IsotropicScalingPlusSymmetricLowRank(
    CongruenceTransform
):
    pass


@linox.congruence_transform.dispatch
def _(
    A: linox.LinearOperator,
    B: linox.IsotropicScalingPlusSymmetricLowRank,
) -> CongruenceTransform_LinearOperator_IsotropicScalingPlusSymmetricLowRank:
    return CongruenceTransform_LinearOperator_IsotropicScalingPlusSymmetricLowRank(A, B)


@linox.diagonal.dispatch
def _(
    A: CongruenceTransform_LinearOperator_IsotropicScalingPlusSymmetricLowRank,
) -> jax.Array:
    return sum(
        linox.diagonal(linox.congruence_transform(A._A, summand))
        for summand in A._B.operator_list
    )


##########################################################
# (LinearOperator, PositiveDiagonalPlusSymmetricLowRank) #
##########################################################


class CongruenceTransform_LinearOperator_PositiveDiagonalPlusSymmetricLowRank(
    CongruenceTransform
):
    pass


@linox.congruence_transform.dispatch
def _(
    A: linox.LinearOperator,
    B: linox.PositiveDiagonalPlusSymmetricLowRank,
) -> CongruenceTransform_LinearOperator_PositiveDiagonalPlusSymmetricLowRank:
    return CongruenceTransform_LinearOperator_PositiveDiagonalPlusSymmetricLowRank(A, B)


@linox.diagonal.dispatch
def _(
    A: CongruenceTransform_LinearOperator_PositiveDiagonalPlusSymmetricLowRank,
) -> jax.Array:
    return sum(
        linox.diagonal(linox.congruence_transform(A._A, summand))
        for summand in A._B.operator_list
    )


##########################################
# (LinearOperator, ScaledLinearOperator) #
##########################################


@linox.congruence_transform.dispatch
def _(
    A: linox.LinearOperator,
    B: ScaledLinearOperator,
) -> ScaledLinearOperator:
    return B.scalar * linox.congruence_transform(A, B.operator)


######################################
# (LinearOperator, SymmetricLowRank) #
######################################


@linox.congruence_transform.dispatch
def _(A: linox.LinearOperator, B: linox.SymmetricLowRank) -> linox.SymmetricLowRank:
    return linox.SymmetricLowRank(A @ B.U, B.S)
