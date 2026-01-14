import numpy as np

from state import State, System


class Integrator:
    """
    Abstract base class for toy engine integrators.
    """

    def step(self, system: System, state: State) -> State:
        """
        Take an MD step. Update the state.

        Parameters
        ----------
        system : :class:`System`
            contains force field and particle masses
        state : :class:`State`
            positions and velocities for the particles
        """
        raise NotImplementedError("step() must be implemented in subclass")

    def _position_update(self, system: System, state: State, dt: float) -> State:
        new_positions = state.positions + state.velocities * dt
        return State(positions=new_positions, velocities=state.velocities)

    def _velocity_update(self, system: System, state: State, dt: float) -> State:
        forces = system.forcefield.compute_forces(state.positions)
        new_velocities = state.velocities + forces * system.inv_masses[:, np.newaxis] * dt
        return State(positions=state.positions, velocities=new_velocities)


class EulerIntegrator(Integrator):
    """Euler integrator

    Parameters
    ----------
    dt : float
        time step
    """

    def __init__(self, dt: float):
        self.dt = dt

    def step(self, system: System, state: State) -> State:
        """
        Take an MD step. Update the state.
        """
        state = self._velocity_update(system, state, self.dt)
        state = self._position_update(system, state, self.dt)
        return state


class VelocityVerletIntegrator(Integrator):
    """Velocity Verlet integrator

    Parameters
    ----------
    dt : float
        time step
    """

    def __init__(self, dt: float):
        self.dt = dt

    def step(self, system: System, state: State) -> State:
        """
        Take an MD step. Update the state.
        {
        Vec3d new_pos = pos + vel*dt + acc*(dt*dt*0.5);
        Vec3d new_acc = apply_forces();
        Vec3d new_vel = vel + (acc+new_acc)*(dt*0.5);
        pos = new_pos;
        vel = new_vel;
        acc = new_acc;
        }
        """
        # TODO: Implemement how we update the state
        state = self._velocity_update(system, state, self.dt * 0.5)
        state = self._position_update(system, state, self.dt)
        state = self._velocity_update(system, state, self.dt * 0.5)
        return state

class LangevinIntegratorBase(Integrator):
    """Base class for Langevin integrators

    Parameters
    ----------
    dt : float
        time step
    temperature : float
        temperature in Kelvin
    friction_coeff : float
        friction coefficient in 1/ps
    """

    kB = 0.0019872041  # Boltzmann constant in kcal/(mol*K)

    def __init__(self, dt: float, temperature: float, friction_coeff: float):
        self.dt = dt
        self.temperature = temperature
        self.friction_coeff = friction_coeff

    def _random_force(self, system: System) -> np.ndarray:
        return np.random.normal(0.0, 1.0, size=(system.num_particles, 2))


class LangevinBAOABIntegrator(LangevinIntegratorBase):
    """Langevin BAOAB integrator

    Parameters
    ----------
    dt : float
        time step
    temperature : float
        temperature in Kelvin
    friction_coeff : float
        friction coefficient in 1/ps
    """

    def __init__(self, dt: float, temperature: float, friction_coeff: float):
        super().__init__(dt, temperature, friction_coeff)
        self.c1 = np.exp(-friction_coeff * dt)
        self.c2 = (1.0 - self.c1) / friction_coeff

    def _OU_update(self, system: System, state: State) -> State:
        random_force = self._random_force(system)
        c3 = np.sqrt((1.0 - self.c1 ** 2) * self.kB * self.temperature / system.masses)
        new_velocities = (self.c1 * state.velocities +
                          c3[:, np.newaxis] * random_force)
        return State(positions=state.positions, velocities=new_velocities)

    def step(self, system: System, state: State) -> State:
        """
        Take an MD step. Update the state.
        """
        state = self._velocity_update(system, state, 0.5 * self.dt)
        state = self._position_update(system, state, 0.5 * self.dt)
        state = self._OU_update(system, state)
        state = self._position_update(system, state, 0.5 * self.dt)
        state = self._velocity_update(system, state, 0.5 * self.dt)
        return state


class VVVRIntegrator(LangevinIntegratorBase):
    """VVVR integrator

    Parameters
    ----------
    dt : float
        time step
    temperature : float
        temperature in Kelvin
    friction_coeff : float
        friction coefficient in 1/ps
    """

    def __init__(self, dt: float, temperature: float, friction_coeff: float):
        super().__init__(dt, temperature, friction_coeff)

    def _OU_update(self, system: System, state: State) -> State:
        random_force = self._random_force(system)
        c1 = np.exp(-self.friction_coeff * self.dt)
        c2 = np.sqrt((1.0 - c1 ** 2) * self.kB * self.temperature / system.masses)
        new_velocities = (c1 * state.velocities +
                          c2[:, np.newaxis] * random_force)
        return State(positions=state.positions, velocities=new_velocities)

    def step(self, system: System, state: State) -> State:
        """
        Take an MD step. Update the state.
        """
        state = self._velocity_update(system, state, 0.5 * self.dt)
        state = self._position_update(system, state, self.dt)
        state = self._OU_update(system, state)
        state = self._velocity_update(system, state, 0.5 * self.dt)
        return state
