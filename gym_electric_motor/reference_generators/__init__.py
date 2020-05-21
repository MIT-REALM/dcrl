from .wiener_process_reference_generator import WienerProcessReferenceGenerator
from .switched_reference_generator import SwitchedReferenceGenerator
from .zero_reference_generator import ZeroReferenceGenerator
from .sinusoidal_reference_generator import SinusoidalReferenceGenerator
from .step_reference_generator import StepReferenceGenerator
from .triangle_reference_generator import TriangularReferenceGenerator
from .square_reference_generator import SquareReferenceGenerator
from .sawtooth_reference_generator import SawtoothReferenceGenerator
from .const_reference_generator import ConstReferenceGenerator
from .multiple_reference_generator import MultipleReferenceGenerator
from ..utils import register_class
from ..core import ReferenceGenerator

register_class(WienerProcessReferenceGenerator, ReferenceGenerator, 'WienerProcessReference')
register_class(SwitchedReferenceGenerator, ReferenceGenerator, 'SwitchedReference')
register_class(StepReferenceGenerator, ReferenceGenerator, 'StepReference')
register_class(SinusoidalReferenceGenerator, ReferenceGenerator, 'SinusReference')
register_class(TriangularReferenceGenerator, ReferenceGenerator, 'TriangleReference')
register_class(SquareReferenceGenerator, ReferenceGenerator, 'SquareReference')
register_class(SawtoothReferenceGenerator, ReferenceGenerator, 'SawtoothReference')
register_class(ConstReferenceGenerator, ReferenceGenerator, 'ConstReference')
register_class(MultipleReferenceGenerator, ReferenceGenerator, 'MultipleReference')
