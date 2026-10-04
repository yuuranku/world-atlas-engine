"""Conservative box ranges and directional derivatives of the actual ground.

The oracle evaluates the existing refinement formula with outward interval
arithmetic. It never reconstructs the refined field from native samples.
At piecewise-smooth switches, derivative intervals include every possible
branch, so a strictly signed directional interval certifies monotonicity.
"""
import numpy as np
import shapely

from .continuous_terrain import PhysicalTerrainField
from .terrain_refinement import RefinedTerrainField


def _down(value):
    return np.nextafter(value, -np.inf)


def _up(value):
    return np.nextafter(value, np.inf)


def _product(first, last):
    a, b = first
    c, d = last
    ac, ad, bc, bd = a*c, a*d, b*c, b*d
    low = np.minimum(np.minimum(ac, ad), np.minimum(bc, bd))
    high = np.maximum(np.maximum(ac, ad), np.maximum(bc, bd))
    exact_zero = ((a==0)&(b==0))|((c==0)&(d==0))
    return np.where(exact_zero,0,_down(low)),np.where(exact_zero,0,_up(high))


class _Dual:
    __array_priority__ = 1000

    def __init__(self, low, high=None, gradient_low=None, gradient_high=None):
        self.low = np.asarray(low, dtype=float)
        self.high = self.low.copy() if high is None else np.asarray(high, dtype=float)
        shape = self.low.shape+(2,)
        self.gradient_low = np.zeros(shape) if gradient_low is None else np.asarray(gradient_low)
        self.gradient_high = self.gradient_low.copy() if gradient_high is None else np.asarray(gradient_high)

    def __getitem__(self, index):
        return _Dual(self.low[index], self.high[index], self.gradient_low[index], self.gradient_high[index])

    def __neg__(self):
        return _Dual(-self.high, -self.low, -self.gradient_high, -self.gradient_low)

    def _constant(self,value):
        return ((self.low==value)&(self.high==value)
                &np.all((self.gradient_low==0)&(self.gradient_high==0),axis=-1))

    def __add__(self, other):
        other = other if isinstance(other, _Dual) else _Dual(other)
        low,high = _value_add((self.low,self.high),(other.low,other.high))
        gl,gh = _value_add((self.gradient_low,self.gradient_high),
                           (other.gradient_low,other.gradient_high))
        for constant,value in ((self,other),(other,self)):
            exact = constant._constant(0)
            low,high = np.where(exact,value.low,low),np.where(exact,value.high,high)
            gl,gh = np.where(exact[...,None],value.gradient_low,gl),np.where(exact[...,None],value.gradient_high,gh)
        return _Dual(low,high,gl,gh)

    __radd__ = __add__

    def __sub__(self, other):
        other = other if isinstance(other, _Dual) else _Dual(other)
        result = self+(-other)
        exact = (self.low==self.high)&(other.low==other.high)&(self.low==other.low)
        result.low,result.high = np.where(exact,0,result.low),np.where(exact,0,result.high)
        exact_gradient = ((self.gradient_low==self.gradient_high)
                          &(other.gradient_low==other.gradient_high)
                          &(self.gradient_low==other.gradient_low))
        result.gradient_low = np.where(exact_gradient,0,result.gradient_low)
        result.gradient_high = np.where(exact_gradient,0,result.gradient_high)
        return result

    def __rsub__(self, other):
        return (-self)+other

    def __mul__(self, other):
        other = other if isinstance(other, _Dual) else _Dual(other)
        low, high = _product((self.low,self.high),(other.low,other.high))
        first = _product((self.gradient_low,self.gradient_high),
                         (other.low[...,None],other.high[...,None]))
        last = _product((other.gradient_low,other.gradient_high),
                        (self.low[...,None],self.high[...,None]))
        gl,gh = _value_add(first,last)
        for constant,value in ((self,other),(other,self)):
            exact_value = (constant.low==1)&(constant.high==1)
            low,high = np.where(exact_value,value.low,low),np.where(exact_value,value.high,high)
            exact = constant._constant(1)
            gl,gh = np.where(exact[...,None],value.gradient_low,gl),np.where(exact[...,None],value.gradient_high,gh)
        exact_zero = self._constant(0)|other._constant(0)
        return _Dual(np.where(exact_zero,0,low),np.where(exact_zero,0,high),
                     np.where(exact_zero[...,None],0,gl),np.where(exact_zero[...,None],0,gh))

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, _Dual) else _Dual(other)
        if np.any((other.low <= 0)&(other.high >= 0)):
            raise ValueError('terrain interval division must exclude zero')
        minimum = np.minimum(abs(other.low),abs(other.high))
        maximum = np.maximum(abs(other.low),abs(other.high))
        gradient = _product((other.gradient_low,other.gradient_high),
                            (_down(-1/minimum**2)[...,None],_up(-1/maximum**2)[...,None]))
        return self*_Dual(_down(1/other.high),_up(1/other.low),*gradient)

    def __rtruediv__(self, other):
        return _Dual(other)/self

    def square(self):
        low = np.where((self.low <= 0)&(self.high >= 0),0,
                       np.minimum(self.low**2,self.high**2))
        high = np.maximum(self.low**2,self.high**2)
        gradient = _product((self.gradient_low,self.gradient_high),
                            (_down(2*self.low)[...,None],_up(2*self.high)[...,None]))
        return _Dual(np.maximum(0,_down(low)),_up(high),*gradient)

    def power(self, exponent):
        if exponent == 2:
            return self.square()
        result = self
        for _ in range(1,exponent):
            result = result*self
        return result


def _clip(value, low, high):
    outside = (value.high <= low)|(value.low >= high)
    inside = (value.low >= low)&(value.high <= high)
    first = np.where(inside[...,None],value.gradient_low,np.minimum(0,value.gradient_low))
    last = np.where(inside[...,None],value.gradient_high,np.maximum(0,value.gradient_high))
    return _Dual(np.clip(value.low,low,high),np.clip(value.high,low,high),
                 np.where(outside[...,None],0,first),np.where(outside[...,None],0,last))


def _absolute(value):
    positive, negative = value.low > 0, value.high < 0
    gradient = _product((value.gradient_low,value.gradient_high),
                        (np.where(positive,1,-1)[...,None],np.where(negative,-1,1)[...,None]))
    return _Dual(np.where(positive,value.low,np.where(negative,-value.high,0)),
                 np.maximum(abs(value.low),abs(value.high)),*gradient)


def _tanh(value):
    low, high = _down(np.tanh(value.low)),_up(np.tanh(value.high))
    maximum = np.maximum(abs(low),abs(high))
    minimum = np.where((low <= 0)&(high >= 0),0,np.minimum(abs(low),abs(high)))
    factor = (np.maximum(0,_down(1-maximum**2)),np.minimum(1,_up(1-minimum**2)))
    gradient = _product((value.gradient_low,value.gradient_high),
                        (factor[0][...,None],factor[1][...,None]))
    zero = (value.low==0)&(value.high==0)
    return _Dual(np.where(zero,0,low),np.where(zero,0,high),
                 np.where(zero[...,None],value.gradient_low,gradient[0]),
                 np.where(zero[...,None],value.gradient_high,gradient[1]))


def _slope(first,last):
    """Uniform PCHIP's actual harmonic branch, including all switch derivatives."""
    def harmonic(a,b):
        active = ((a > 0)&(b > 0))|((a < 0)&(b < 0))
        # Match SciPy's uniform-grid weighted harmonic calculation and bound
        # each intermediate operation, including its binary64 representation.
        a,b = np.where(active,a,1),np.where(active,b,1)
        with np.errstate(over='ignore', divide='ignore'):
            ra,rb = 3/a,3/b
            mean_low = _down(_down(_down(ra)+_down(rb))/6)
            mean_high = _up(_up(_up(ra)+_up(rb))/6)
            low,high = _down(1/mean_high),_up(1/mean_low)
        low = np.where(a > 0,np.maximum(0,low),low)
        high = np.where(a < 0,np.minimum(0,high),high)
        return np.where(active,low,0),np.where(active,high,0)
    low, high = harmonic(first.low,last.low)[0],harmonic(first.high,last.high)[1]
    regular = ((first.low > 0)&(last.low > 0))|((first.high < 0)&(last.high < 0))
    inactive = ((first.high < 0)&(last.low > 0))|((first.low > 0)&(last.high < 0))
    dl, dh = _down(first.low+last.low),_up(first.high+last.high)
    denominator_regular = (dl > 0)|(dh < 0)
    minimum = np.where(denominator_regular,np.minimum(abs(dl),abs(dh)),1)
    maximum = np.where(denominator_regular,np.maximum(abs(dl),abs(dh)),1)
    factors = []
    for value in (last,first):
        amin = np.where((value.low <= 0)&(value.high >= 0),0,np.minimum(abs(value.low),abs(value.high)))
        amax = np.maximum(abs(value.low),abs(value.high))
        factors.append((np.where(inactive,0,np.where(regular,np.maximum(0,_down(2*amin**2/maximum**2)),0)),
                        np.where(inactive,0,np.where(denominator_regular,
                            np.minimum(2,_up(2*amax**2/minimum**2)),2))))
    a = _product((first.gradient_low,first.gradient_high),
                 (factors[0][0][...,None],factors[0][1][...,None]))
    b = _product((last.gradient_low,last.gradient_high),
                 (factors[1][0][...,None],factors[1][1][...,None]))
    return _Dual(low,high,*_value_add(a,b))


def _hermite(values,position,differences):
    first,last = values[1],values[2]
    before = _slope(differences[0],differences[1])
    after = _slope(differences[1],differences[2])
    delta = differences[1]
    cubic = before+after-2*delta
    quadratic = delta-before-cubic
    result = ((cubic*position+quadratic)*position+before)*position+first
    # Collect the repeated slope terms before differentiating. Differentiating
    # their separate Horner appearances loses the dependency between +m, -2m
    # and +m, and invents opposite x gradients at a genuine PCHIP slope switch.
    # All three Hermite basis functions use the same abscissa. Evaluate their
    # unchanged interval polynomials together instead of repeating the full
    # Bernstein and derivative construction three times.
    coefficients = np.asarray(((-2.,3.,0.,0.),(1.,-2.,1.,0.),(1.,-1.,0.,0.))).T
    coefficients = np.broadcast_to(coefficients[:,:,None],(4,3,len(position.low)))
    parameter = _Dual(position.low[None,:],position.high[None,:],
                      position.gradient_low[None,:,:],position.gradient_high[None,:,:])
    values = _polynomial(coefficients,parameter)
    values.low = np.maximum(np.asarray((0.,0.,_down(-4/27)))[:,None],values.low)
    values.high = np.minimum(np.asarray((1.,_up(4/27),0.))[:,None],values.high)
    weights = tuple(values[index] for index in range(3))
    gradient = first.gradient_low,first.gradient_high
    for value,weight in zip((delta,before,after),weights,strict=True):
        gradient = _value_add(gradient,_product((value.gradient_low,value.gradient_high),
                                                (weight.low[:,None],weight.high[:,None])))
        gradient = _value_add(gradient,_product((weight.gradient_low,weight.gradient_high),
                                                (value.low[:,None],value.high[:,None])))
    return _Dual(result.low,result.high,*gradient)


def _value_add(first,last):
    a,b = first;c,d = last
    low,high = _down(a+c),_up(b+d)
    first_zero,last_zero = (a==0)&(b==0),(c==0)&(d==0)
    return (np.where(first_zero,c,np.where(last_zero,a,low)),
            np.where(first_zero,d,np.where(last_zero,b,high)))


def _value_subtract(first,last):
    low,high = _value_add(first,(-last[1],-last[0]))
    exact = (first[0]==first[1])&(last[0]==last[1])&(first[0]==last[0])
    return np.where(exact,0,low),np.where(exact,0,high)


def _value_scale(value,factor):
    low,high = _product(value,(np.asarray(factor),np.asarray(factor)))
    return low,high


def _polynomial(coefficients,position):
    """Cubic/derivative Bernstein hulls on the actual abscissa interval."""
    bounds = (list(zip(coefficients.low,coefficients.high,strict=True))
              if isinstance(coefficients,_Dual) else [(v,v) for v in coefficients])
    first,last = (position.low,position.low),(position.high,position.high)
    width = _value_subtract(last,first)
    def evaluate(t):
        value=bounds[0]
        for coefficient in bounds[1:]:
            value=_value_add(_product(value,t),coefficient)
        return value
    def derivative(t):
        return _value_add(_product(_value_add(_product(_value_scale(bounds[0],3),t),
                                               _value_scale(bounds[1],2)),t),bounds[2])
    a,b,da,db=evaluate(first),evaluate(last),derivative(first),derivative(last)
    def third(value):
        return np.where(value[0]==0,0,_down(value[0]/3)),np.where(value[1]==0,0,_up(value[1]/3))
    controls=(a,_value_add(a,third(_product(width,da))),
              _value_subtract(b,third(_product(width,db))),b)
    low=np.minimum.reduce([v[0]for v in controls]);high=np.maximum.reduce([v[1]for v in controls])
    middle=_value_add(da,_product(width,_value_add(_product(_value_scale(bounds[0],3),first),bounds[1])))
    derivative_controls=(da,middle,db)
    dl=np.minimum.reduce([v[0]for v in derivative_controls]);dh=np.maximum.reduce([v[1]for v in derivative_controls])
    gradient = _product((position.gradient_low,position.gradient_high),(dl[...,None],dh[...,None]))
    return _Dual(low,high,*gradient)


def _polynomial_roundoff(coefficients,position):
    # Six operations evaluate a horizontal cubic in the actual model. Keep
    # their rounding allowance when cancelling common polynomial terms before
    # the vertical PCHIP; this is a scale-dependent IEEE error bound, not a
    # geometric tolerance.
    t = np.maximum(abs(position.low),abs(position.high))
    magnitude = abs(coefficients[0])
    for coefficient in coefficients[1:]:
        magnitude = _up(_up(magnitude*t)+abs(coefficient))
    gamma = 6*np.finfo(float).eps/(1-6*np.finfo(float).eps)
    constant = np.all(coefficients[:3]==0,axis=0)
    return np.where(constant,0,_up(gamma*magnitude))


def _restrict(value,indices,low,high):
    return _Dual(np.maximum(value.low[indices],low),np.minimum(value.high[indices],high),
                 value.gradient_low[indices],value.gradient_high[indices])


def _raster(native,x,y,*,horizontal_coefficients):
    """Union every intersected native coefficient patch after the true warp."""
    height,width = native.shape
    column,row = np.floor(x.low-.5).astype(int),np.floor(y.low-.5).astype(int)
    final_column,final_row = np.floor(x.high-.5).astype(int),np.floor(y.high-.5).astype(int)
    low,high = np.full(len(column),np.inf),np.full(len(column),-np.inf)
    gl,gh = np.full((len(column),2),np.inf),np.full((len(column),2),-np.inf)
    for dy in range(int(np.max(final_row-row))+1):
        rr = row+dy
        for dx in range(int(np.max(final_column-column))+1):
            cc = column+dx
            indices = np.flatnonzero((cc <= final_column)&(rr <= final_row))
            if not len(indices):
                continue
            xx = _restrict(x,indices,cc[indices]+.5,cc[indices]+1.5)-(cc[indices]+.5)
            yy = _restrict(y,indices,rr[indices]+.5,rr[indices]+1.5)-(rr[indices]+.5)
            xx.low,xx.high = np.maximum(0,xx.low),np.minimum(1,xx.high)
            yy.low,yy.high = np.maximum(0,yy.low),np.minimum(1,yy.high)
            if horizontal_coefficients is None:
                a,b = native[np.clip(rr[indices],0,height-1),(cc[indices])%width],native[np.clip(rr[indices],0,height-1),(cc[indices]+1)%width]
                c,d = native[np.clip(rr[indices]+1,0,height-1),(cc[indices])%width],native[np.clip(rr[indices]+1,0,height-1),(cc[indices]+1)%width]
                result = (1-yy)*((1-xx)*a+xx*b)+yy*((1-xx)*c+xx*d)
            else:
                rows = np.clip(rr[indices][None,:]+np.arange(-1,3)[:,None],0,height-1)
                polynomials = horizontal_coefficients[:,rows,cc[indices]%width]
                roundoff = _polynomial_roundoff(polynomials,xx)
                values = _polynomial(polynomials,xx)
                # Rows share one abscissa. Batch their three differences while
                # retaining the same outward rounding at every operation.
                differences = _polynomial(_Dual(polynomials[:,1:])-_Dual(polynomials[:,:-1]),xx)
                allowance = roundoff[1:]+roundoff[:-1]
                rounded = allowance!=0
                allowance = np.where(rounded,_up(allowance),0)
                differences.low = np.where(rounded,_down(differences.low-allowance),differences.low)
                differences.high = np.where(rounded,_up(differences.high+allowance),differences.high)
                rows = [values[j] for j in range(4)]
                result = _hermite(rows,yy,
                                  [differences[j] for j in range(3)])
                # Shape-preserving vertical interpolation stays between its
                # two containing horizontal rows. Their intervals already
                # include the source polynomial's complete evaluation error.
                # The shape-preserving formula is exact in real arithmetic;
                # SciPy constructs and evaluates its coefficients in binary64.
                # Thirty-two operations cover both harmonic slopes, coefficient
                # construction and Horner evaluation, including underflow.
                magnitude = np.zeros(len(indices))
                for row_values in rows:
                    magnitude = _up(magnitude+np.maximum(abs(row_values.low),abs(row_values.high)))
                gamma = 32*np.finfo(float).eps/(1-32*np.finfo(float).eps)
                allowance = _up(gamma*magnitude+32*np.nextafter(0.,1.))
                result.low = np.maximum(result.low,_down(np.minimum(rows[1].low,rows[2].low)-allowance))
                result.high = np.minimum(result.high,_up(np.maximum(rows[1].high,rows[2].high)+allowance))
            low[indices] = np.minimum(low[indices],result.low)
            high[indices] = np.maximum(high[indices],result.high)
            gl[indices] = np.minimum(gl[indices],result.gradient_low)
            gh[indices] = np.maximum(gh[indices],result.gradient_high)
    return _Dual(low,high,gl,gh)


def _sum_groups(size,owner,value):
    """Outward sum, including the IEEE rounding of sparse accumulation."""
    count = np.bincount(owner,minlength=size)
    gamma = count*np.finfo(float).eps/(1-count*np.finfo(float).eps)
    def accumulate(first,last):
        shape = (size,)+first.shape[1:]
        low,high,magnitude = np.zeros(shape),np.zeros(shape),np.zeros(shape)
        np.add.at(low,owner,first);np.add.at(high,owner,last)
        np.add.at(magnitude,owner,np.maximum(abs(first),abs(last)))
        factor = (gamma/(1-gamma)).reshape((size,)+(1,)*(first.ndim-1))
        allowance = _up(factor*magnitude)
        return np.where(magnitude==0,0,_down(low-allowance)),np.where(magnitude==0,0,_up(high+allowance))
    low,high = accumulate(value.low,value.high)
    gl,gh = accumulate(value.gradient_low,value.gradient_high)
    return _Dual(low,high,gl,gh)


def _minimum(values):
    high = np.minimum.reduce([value.high for value in values])
    low = np.minimum.reduce([value.low for value in values])
    gl,gh = np.full(high.shape+(2,),np.inf),np.full(high.shape+(2,),-np.inf)
    for value in values:
        possible = value.low <= high
        gl = np.minimum(gl,np.where(possible[...,None],value.gradient_low,np.inf))
        gh = np.maximum(gh,np.where(possible[...,None],value.gradient_high,-np.inf))
    return _Dual(np.maximum(0,low),np.maximum(0,high),gl,gh)


def _minimum_groups(size,owner,value):
    low,high = np.full(size,np.inf),np.full(size,np.inf)
    np.minimum.at(low,owner,value.low);np.minimum.at(high,owner,value.high)
    possible = value.low <= high[owner]
    gl,gh = np.full((size,2),np.inf),np.full((size,2),-np.inf)
    np.minimum.at(gl,owner[possible],value.gradient_low[possible])
    np.maximum.at(gh,owner[possible],value.gradient_high[possible])
    return _Dual(np.maximum(0,low),np.maximum(0,high),gl,gh)


def _segment_distance_squared(x,y,start,end,east=1.,north=1.,*,denominator_floor=0.):
    ax,ay = (start[...,0]-x)*east,(start[...,1]-y)*north
    dx,dy = _Dual(end[...,0]-start[...,0])*east,_Dual((end[...,1]-start[...,1])*north)
    denominator = dx.square()+dy.square()
    if denominator_floor:
        denominator = _clip(denominator,denominator_floor,np.inf)
    along = _clip(-(ax*dx+ay*dy)/denominator,0,1)
    result = (ax+along*dx).square()+(ay+along*dy).square()
    return _Dual(np.maximum(0,result.low),result.high,result.gradient_low,result.gradient_high)


def _distance_weight(distance_squared,first,last):
    dl,dh = np.sqrt(np.maximum(0,distance_squared.low)),np.sqrt(distance_squared.high)
    tlo,thi = np.clip((dl-first)/(last-first),0,1),np.clip((dh-first)/(last-first),0,1)
    low,high = tlo*tlo*(3-2*tlo),thi*thi*(3-2*thi)
    active = (dh > first)&(dl < last)
    a,b = np.maximum(dl,first),np.minimum(dh,last)
    t0,t1 = np.clip((a-first)/(last-first),0,1),np.clip((b-first)/(last-first),0,1)
    minimum = np.minimum(t0*(1-t0),t1*(1-t1))
    maximum = np.where((t0 <= .5)&(t1 >= .5),.25,np.maximum(t0*(1-t0),t1*(1-t1)))
    fl = np.where(active,3*minimum/((last-first)*np.maximum(b,first)),0)
    fh = np.where(active,3*maximum/((last-first)*np.maximum(a,first)),0)
    fl = np.where((dl < first)|(dh > last),0,fl)
    gl,gh = _product((distance_squared.gradient_low,distance_squared.gradient_high),
                     (np.maximum(0,_down(fl))[:,None],_up(fh)[:,None]))
    outside = ~active
    gl[outside],gh[outside] = 0,0
    low,high = np.maximum(0,_down(low)),np.minimum(1,_up(high))
    zero,one = dh<=first,dl>=last
    return _Dual(np.where(zero,0,np.where(one,1,low)),
                 np.where(zero,0,np.where(one,1,high)),gl,gh)


def _protection(field,x,y):
    result = _Dual(np.ones(len(x.low)))
    if field._anchors is not None:
        centres = np.column_stack(((x.low+x.high)/2,(y.low+y.high)/2))
        radius = np.hypot(x.high-x.low,y.high-y.low)/2+.5
        candidates = field._anchors.query_ball_point(centres,radius)
        owners = np.repeat(np.arange(len(centres)),[len(item) for item in candidates])
        if len(owners):
            anchors = field._anchors.data[np.concatenate(candidates).astype(int)]
            distances = (x[owners]-anchors[:,0]).square()+(y[owners]-anchors[:,1]).square()
            present = np.unique(owners)
            grouped = _minimum_groups(len(centres),owners,distances)
            weight = _Dual(np.ones(len(centres)),gradient_low=np.zeros((len(centres),2)))
            local = _distance_weight(grouped[present],.1,.5)
            weight.low[present],weight.high[present] = local.low,local.high
            weight.gradient_low[present],weight.gradient_high[present] = local.gradient_low,local.gradient_high
            result = result*weight
    if field._rivers is not None:
        pairs = field._rivers.query(shapely.box(x.low-.5,y.low-.5,x.high+.5,y.high+.5))
        if pairs.shape[1]:
            owner,edge = pairs
            segments = shapely.get_coordinates(field._rivers.geometries[edge]).reshape(-1,2,2)
            distances = _segment_distance_squared(x[owner],y[owner],segments[:,0],segments[:,1])
            grouped = _minimum_groups(len(x.low),owner,distances)
            present = np.unique(owner)
            weight = _Dual(np.ones(len(x.low)),gradient_low=np.zeros((len(x.low),2)))
            local = _distance_weight(grouped[present],.18,.5)
            weight.low[present],weight.high[present] = local.low,local.high
            weight.gradient_low[present],weight.gradient_high[present] = local.gradient_low,local.gradient_high
            result = result*weight
    return result


def _warp(field,x,y):
    patches = field._landforms
    pairs = patches._tree.query(shapely.box(x.low,y.low,x.high,y.high))
    if not pairs.shape[1]:
        return x,y
    owner,patch = pairs
    squared = ((x[owner]-patches.centres[patch,0]).square()
               +(y[owner]-patches.centres[patch,1]).square())/(patches.radius[patch]**2)
    influence = _clip(1-squared,0,1).power(3)
    return (x+_sum_groups(len(x.low),owner,influence*patches.displacement[patch,0]),
            y+_sum_groups(len(x.low),owner,influence*patches.displacement[patch,1]))


def _east_metric(drainage,y):
    angle = np.pi/2-(y*np.pi)/drainage.height
    low = np.minimum(np.cos(angle.low),np.cos(angle.high))
    high = np.where((angle.low <= 0)&(angle.high >= 0),1,
                    np.maximum(np.cos(angle.low),np.cos(angle.high)))
    derivative = _product((angle.gradient_low,angle.gradient_high),
                          (_down(-np.sin(angle.high))[:,None],_up(-np.sin(angle.low))[:,None]))
    return _clip(_Dual(_down(low),_up(high),*derivative),.08,np.inf)*drainage.cell_x_km


def _drainage(drainage,x,y):
    if not drainage.edge_count:
        return _Dual(np.zeros(len(x.low)))
    east = _east_metric(drainage,y)
    margin_x = drainage.maximum_breadth/east.low
    margin_y = drainage.maximum_breadth/drainage.cell_y_km
    pairs = drainage._tree.query(shapely.box(x.low-margin_x,y.low-margin_y,
                                            x.high+margin_x,y.high+margin_y))
    if not pairs.shape[1]:
        return _Dual(np.zeros(len(x.low)))
    owner,edge = pairs
    curve = drainage.curves[edge]
    # Every compact valley uses four line pieces. Their distance formula and
    # directional arithmetic are independent and share x/y intervals, so one
    # array batch preserves all four ownership candidates without four Python
    # evaluations of the same expression tree.
    distances = _segment_distance_squared(x[owner][None,:],y[owner][None,:],
        curve[:,:-1].transpose(1,0,2),curve[:,1:].transpose(1,0,2),
        east[owner][None,:],drainage.cell_y_km,denominator_floor=1e-15)
    squared = _minimum([distances[index] for index in range(4)])
    breadth = drainage.breadth[edge]
    first = np.clip(np.sqrt(squared.low)/breadth,0,1)
    last = np.clip(np.sqrt(squared.high)/breadth,0,1)
    low,high = (1-last)**4*(1+4*last),(1-first)**4*(1+4*first)
    factor = (_down(-10*(1-first)**3/breadth**2),_up(-10*(1-last)**3/breadth**2))
    gl,gh = _product((squared.gradient_low,squared.gradient_high),(factor[0][:,None],factor[1][:,None]))
    inactive = first >= 1
    gl[inactive],gh[inactive] = 0,0
    contribution = _Dual(np.where(inactive,0,np.maximum(0,_down(low))),
                         np.where(inactive,0,np.minimum(1,_up(high))),gl,gh)*drainage.strength[edge]
    return -_tanh(_sum_groups(len(x.low),owner,contribution))


def _evaluate(field,lower,upper,*,directions=None):
    if (isinstance(field, RefinedTerrainField) and field._river_bed is not None
            and len(field._river_bed.segments)):
        raise ValueError("interval terrain certificates do not support the continuous river-bed model")
    count = len(lower)
    if directions is None:
        dx,dy = np.tile([1.,0.],(count,1)),np.tile([0.,1.],(count,1))
    else:
        dx,dy = directions[:,0,:],directions[:,1,:]
    x,y = _Dual(lower[:,0],upper[:,0],dx),_Dual(lower[:,1],upper[:,1],dy)
    if isinstance(field,PhysicalTerrainField):
        return _raster(field.native_m,x,y,horizontal_coefficients=field.horizontal_coefficients)
    warped_x,warped_y = _warp(field,x,y)
    ground = _raster(field.base.native_m,warped_x,warped_y,
                     horizontal_coefficients=field.base.horizontal_coefficients)
    detail = _drainage(field._height,x,y)-_raster(field._height.pchip.native_m,x,y,
                     horizontal_coefficients=field._height.pchip.horizontal_coefficients)
    amplitude = _raster(field._amplitude,x,y,horizontal_coefficients=None)*_protection(field,x,y)
    gain = 1+.45*_tanh(amplitude/(_absolute(ground)+30))*detail
    result = ground*gain
    # The model explicitly pins native centres, including their binary64
    # representation. Include any such point in a box's range without moving
    # the continuous field or pretending it defines the rest of the box.
    column,row = np.ceil(lower[:,0]-.5).astype(int),np.ceil(lower[:,1]-.5).astype(int)
    final_column,final_row = np.floor(upper[:,0]-.5).astype(int),np.floor(upper[:,1]-.5).astype(int)
    for oy in range(max(0,int(np.max(final_row-row))+1)):
        for ox in range(max(0,int(np.max(final_column-column))+1)):
            cc,rr = column+ox,row+oy
            valid = (cc <= final_column)&(rr <= final_row)&(rr >= 0)&(rr < field.height)
            native = field.native_m[np.clip(rr,0,field.height-1),cc%field.width]
            result.low[valid] = np.minimum(result.low[valid],native[valid])
            result.high[valid] = np.maximum(result.high[valid],native[valid])
    return result


def _checked_boxes(field,lower,upper):
    if not isinstance(field,(PhysicalTerrainField,RefinedTerrainField)):
        raise TypeError('implicit terrain bounds require the actual physical field model')
    lower,upper = np.asarray(lower,dtype=float),np.asarray(upper,dtype=float)
    if (lower.ndim != 2 or lower.shape[1:] != (2,) or upper.shape != lower.shape
            or not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper))
            or np.any(lower > upper) or np.any(lower[:,0] < 0) or np.any(upper[:,0] > field.width)
            or np.any(lower[:,1] < 0) or np.any(upper[:,1] > field.height)
            or np.any(upper-lower > 1)):
        raise ValueError('implicit terrain oracle requires finite ordered native-cell boxes')
    return lower,upper


def _enclosed_result(field,lower,upper):
    result = _evaluate(field,lower,upper)
    middle = lower+(upper-lower)/2
    centre = _evaluate(field,middle,middle)
    radius = _up(np.maximum(middle-lower,upper-middle))
    variation = _up(_up(np.maximum(abs(result.gradient_low),
                                  abs(result.gradient_high))*radius).sum(axis=1))
    # The interval derivative is valid on every piecewise-smooth branch.
    # The mean-value enclosure therefore remains valid across their
    # switches, and contracts quadratically on a near-tangent thin edge.
    result.low = np.maximum(result.low,_down(centre.low-variation))
    result.high = np.minimum(result.high,_up(centre.high+variation))
    return result


def field_range_gradient_bounds(field,lower,upper):
    """Return conservative value and x/y derivative intervals for N boxes."""
    lower,upper = _checked_boxes(field,lower,upper)
    low,high = np.empty(len(lower)),np.empty(len(lower))
    gl,gh = np.empty((len(lower),2)),np.empty((len(lower),2))
    for begin in range(0,len(lower),4096):
        stop = min(begin+4096,len(lower))
        result = _enclosed_result(field,lower[begin:stop],upper[begin:stop])
        low[begin:stop],high[begin:stop] = result.low,result.high
        gl[begin:stop],gh[begin:stop] = result.gradient_low,result.gradient_high
    if (np.any(~np.isfinite(low)) or np.any(~np.isfinite(high))
            or np.any(~np.isfinite(gl)) or np.any(~np.isfinite(gh))):
        raise ValueError('implicit terrain bounds must remain finite')
    return low,high,gl,gh


def directional_bounds(field,lower,upper,directions):
    """Bound two source directions, propagating them before branch minima.

    The columns of each 2x2 matrix are native (dx,dy) directions. Supply one
    fixed matrix or an N-by-2-by-2 array. These are derivative certificates,
    not rotated value bounds: the source boxes and field remain unchanged.
    """
    lower,upper = _checked_boxes(field,lower,upper)
    directions = np.asarray(directions,dtype=float)
    if directions.shape==(2,2):
        directions = np.broadcast_to(directions,(len(lower),2,2))
    if (directions.shape!=(len(lower),2,2) or not np.all(np.isfinite(directions))
            or np.any(np.all(directions==0,axis=1))):
        raise ValueError('terrain directions require two finite nonzero native vectors per box')
    low,high = np.empty((len(lower),2)),np.empty((len(lower),2))
    for begin in range(0,len(lower),4096):
        stop = min(begin+4096,len(lower))
        result = _evaluate(field,lower[begin:stop],upper[begin:stop],directions=directions[begin:stop])
        low[begin:stop],high[begin:stop] = result.gradient_low,result.gradient_high
    if np.any(~np.isfinite(low))or np.any(~np.isfinite(high)):
        raise ValueError('implicit directional bounds must remain finite')
    return low,high


def warped_pchip_event_bounds(field,source_lower,source_upper,fixed_source_x):
    """Enclose the unchanged ground on a transported native PCHIP event.

    The N intervals parameterize source y on ``gamma(t)=inverse_W(c,t)``.
    Return physical lower/upper boxes, height ranges, and derivative ranges
    with respect to t. Longitudes remain unwrapped; no inverse coordinate
    raster or second height field is constructed.
    """
    from .terrain_refinement import _PATCH_DERIVATIVE

    if not isinstance(field,RefinedTerrainField):
        raise TypeError('warped PCHIP events require the actual refined ground')
    source_lower,source_upper = np.asarray(source_lower,dtype=float),np.asarray(source_upper,dtype=float)
    fixed_source_x = np.asarray(fixed_source_x,dtype=float)
    if fixed_source_x.ndim==0:
        fixed_source_x = np.broadcast_to(fixed_source_x,source_lower.shape)
    if (source_lower.ndim!=1 or source_upper.shape!=source_lower.shape
            or fixed_source_x.shape!=source_lower.shape
            or not np.all(np.isfinite(source_lower)) or not np.all(np.isfinite(source_upper))
            or not np.all(np.isfinite(fixed_source_x)) or np.any(source_lower>source_upper)
            or np.any(source_lower<0) or np.any(source_upper>field.height)
            or np.any(source_upper-source_lower>1)):
        raise ValueError('warped PCHIP events require ordered source-y native intervals and finite x')
    count = len(source_lower)
    if not count:
        return np.empty((0,2)),np.empty((0,2)),*(np.empty(0)for _ in range(4))

    # Work in one longitude period, then return the same unwrapped chart.
    c = fixed_source_x%field.width
    period = fixed_source_x-c
    middle = source_lower+(source_upper-source_lower)/2
    px,py = field.inverse_ground_coordinates(c,middle)
    point = np.column_stack((px,py))
    basis = np.broadcast_to(np.eye(2),(count,2,2))
    wx,wy = _warp(field,_Dual(px,gradient_low=basis[:,0]),
                        _Dual(py,gradient_low=basis[:,1]))
    ex = _value_subtract((wx.low,wx.high),(c,c))
    ey = _value_subtract((wy.low,wy.high),(middle,middle))
    ex,ey = np.maximum(abs(ex[0]),abs(ex[1])),np.maximum(abs(ey[0]),abs(ey[1]))
    error = np.where((ex==0)&(ey==0),0,_up(np.hypot(ex,ey)))
    patches = field._landforms
    active = np.zeros(count,dtype=bool)
    displacement = np.zeros((count,2))
    kappa = 0.
    if patches.count:
        pairs = patches._tree.query(shapely.box(_down(c),_down(source_lower),
                                                _up(c),_up(source_upper)))
        if pairs.shape[1]:
            owner,patch = pairs
            active[owner] = True
            np.maximum.at(displacement,owner,abs(patches.displacement[patch]))
        # Disjoint compact supports give W=I+d*phi at every point. Bound the
        # actual displacement coefficients, rather than assuming the inverse
        # solver returned a perfect root or imposing an arbitrary radius.
        magnitude = _up(np.hypot(patches.displacement[:,0],patches.displacement[:,1]))
        kappa = float(_up(np.max(_up(magnitude/patches.radius))*_up(_PATCH_DERIVATIVE)))
        if not kappa<1:
            raise ValueError('ground event inversion requires the model-owned contraction bound')
    inverse_bound = _up(1/_down(1-kappa))
    centre_error = np.where(error==0,0,_up(error*inverse_bound))
    half = _up(np.maximum(middle-source_lower,source_upper-middle))
    radius = _up(_up(half*inverse_bound)+centre_error)
    lower,upper = _down(point-radius[:,None]),_up(point+radius[:,None])
    displacement = np.where(displacement==0,0,_up(displacement))
    shifted_lower = np.column_stack((_down(c-displacement[:,0]),
                                     _down(source_lower-displacement[:,1])))
    shifted_upper = np.column_stack((_up(c+displacement[:,0]),
                                     _up(source_upper+displacement[:,1])))
    lower,upper = np.maximum(lower,shifted_lower),np.minimum(upper,shifted_upper)
    lower[:,1],upper[:,1] = np.maximum(lower[:,1],0),np.minimum(upper[:,1],field.height)
    # A deformation disk is mapped onto itself. Source intervals outside all
    # disks are exactly identity charts, including plateaus and the frame.
    lower[~active] = np.column_stack((c[~active],source_lower[~active]))
    upper[~active] = np.column_stack((c[~active],source_upper[~active]))

    # A source-y interval can become slightly taller than one native cell.
    # Split only the oracle boxes; the actual gamma and its root bracket stay
    # intact, and every source polynomial patch is still evaluated by _raster.
    pieces = np.maximum(1,np.ceil(upper-lower).astype(int))
    owners,starts,ends = [],[],[]
    for dy in range(int(pieces[:,1].max())):
        for dx in range(int(pieces[:,0].max())):
            ids = np.flatnonzero((pieces[:,0]>dx)&(pieces[:,1]>dy))
            first = lower[ids]+(upper[ids]-lower[ids])*np.array((dx,dy))/pieces[ids]
            last = lower[ids]+(upper[ids]-lower[ids])*np.array((dx+1,dy+1))/pieces[ids]
            first = np.where(np.array((dx,dy))==0,lower[ids],first)
            last = np.where(np.array((dx+1,dy+1))==pieces[ids],upper[ids],last)
            owners.append(ids);starts.append(first);ends.append(last)
    owner,starts,ends = np.concatenate(owners),np.concatenate(starts),np.concatenate(ends)
    low,high = np.full(count,np.inf),np.full(count,-np.inf)
    derivative_low,derivative_high = np.full(count,np.inf),np.full(count,-np.inf)
    for begin in range(0,len(owner),4096):
        stop = min(begin+4096,len(owner))
        ids,a,b = owner[begin:stop],starts[begin:stop],ends[begin:stop]
        result = _enclosed_result(field,a,b)
        identity = ~active[ids]
        dlow,dhigh = result.gradient_low[:,1].copy(),result.gradient_high[:,1].copy()
        selected = np.flatnonzero(~identity)
        if len(selected):
            n = len(selected)
            x = _Dual(a[selected,0],b[selected,0],np.broadcast_to((1.,0.),(n,2)))
            y = _Dual(a[selected,1],b[selected,1],np.broadcast_to((0.,1.),(n,2)))
            wx,wy = _warp(field,x,y)
            j00 = wx.gradient_low[:,0],wx.gradient_high[:,0]
            j01 = wx.gradient_low[:,1],wx.gradient_high[:,1]
            j11 = wy.gradient_low[:,1],wy.gradient_high[:,1]
            # det(I+d*grad(phi)^T)=trace(J)-1. This identity holds on every
            # disjoint disk and outside them, and avoids spurious quadratic
            # Jacobian dependencies. Its true determinant is at least 1-kappa.
            det = _value_subtract(_value_add(j00,j11),(1.,1.))
            det = np.maximum(det[0],_down(1-kappa)),np.minimum(det[1],_up(1+kappa))
            reciprocal = _down(1/det[1]),_up(1/det[0])
            tx = _product((-j01[1],-j01[0]),reciprocal)
            ty = _product(j00,reciprocal)
            gx = result.gradient_low[selected,0],result.gradient_high[selected,0]
            gy = result.gradient_low[selected,1],result.gradient_high[selected,1]
            dl,dh = _value_add(_product(gx,tx),_product(gy,ty))
            dlow[selected],dhigh[selected] = dl,dh
        np.minimum.at(low,ids,result.low);np.maximum.at(high,ids,result.high)
        np.minimum.at(derivative_low,ids,dlow);np.maximum.at(derivative_high,ids,dhigh)

    # Intersect the box range with the true one-dimensional event mean value
    # bound. Inverse root roundoff is explicitly enclosed at the midpoint.
    centre_lower = _down(point-centre_error[:,None])
    centre_upper = _up(point+centre_error[:,None])
    centre_lower[:,1] = np.maximum(centre_lower[:,1],0)
    centre_upper[:,1] = np.minimum(centre_upper[:,1],field.height)
    centre_lower[~active],centre_upper[~active] = point[~active],point[~active]
    centre = _enclosed_result(field,centre_lower,centre_upper)
    variation = _up(np.maximum(abs(derivative_low),abs(derivative_high))*half)
    low,high = np.maximum(low,_down(centre.low-variation)),np.minimum(high,_up(centre.high+variation))
    if (np.any(~np.isfinite(low)) or np.any(~np.isfinite(high))
            or np.any(~np.isfinite(derivative_low)) or np.any(~np.isfinite(derivative_high))
            or np.any(low>high) or np.any(derivative_low>derivative_high)):
        raise ValueError('warped PCHIP event certificates must remain finite and ordered')
    lower[:,0] = _down(lower[:,0]+period)
    upper[:,0] = _up(upper[:,0]+period)
    return lower,upper,low,high,derivative_low,derivative_high


def field_bounds(field,lower,upper):
    """Return only the convergent model-owned value enclosure."""
    low,high,_gl,_gh = field_range_gradient_bounds(field,lower,upper)
    return low,high
