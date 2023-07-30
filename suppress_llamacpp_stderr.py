import os, sys 


class suppress_stdout_stderr(object):
    """
    Because llama.cpp outputs a lot of diagnostic information, this class is needed. 
    It redirects the output of llama.cpp to /dev/null.
    llama.cpp is C++ based so it requires special handling to suppress the output in Python.
    It is used like this:
    with suppress_stdout_stderr(True):
        # any llama.cpp calls

    Passing in False will not suppress calls, useful for debugging
    """
    
    def __init__(self, suppress=True) -> None:
        self.suppress = suppress
        
    def __enter__(self):
        if not self.suppress: return self

        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_): 
        if not self.suppress: return       
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()