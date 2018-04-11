if bytes is str:  # pragma: no cover
    def isStringType(t):
        return isinstance(t, basestring)

    def isPath(f):
        return isinstance(f, basestring)

else:  # pragma: no cover
    def isStringType(t):
        return isinstance(t, str)

    def isPath(f):
        return isinstance(f, (bytes, str))
