{ pkgs ? import <nixpkgs> {} }:

let
  extraLibs = with pkgs; [
    stdenv.cc.cc.lib
    zlib
    glib
    libGL
    libGLU
    
    libX11
    libXi
    libXrender
    libICE
    libSM
    libxcb
    libXext
    
    fontconfig
    freetype
    libxkbcommon 
    
    tcl
    tk
  ];
in
pkgs.mkShell {
  name = "python-data-science";
  venvDir = "./.venv";

  buildInputs = [
    # Python with Tkinter enabled
    (pkgs.python3.withPackages (ps: [ ps.tkinter ]))
    pkgs.python3Packages.venvShellHook
  ] ++ extraLibs;

  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -U pip setuptools wheel
    
    if [ -f requirements.txt ]; then
      pip install -r requirements.txt
    else
      pip install numpy matplotlib opencv-python jupyterlab ipykernel jupyterlab-vim
    fi
  '';

  postShellHook = ''
    unset SOURCE_DATE_EPOCH
    
    # 2. Point LD_LIBRARY_PATH to all the libs defined above
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath extraLibs}:$LD_LIBRARY_PATH
    
    # 3. NEW: Tell Qt/OpenCV where to find system fonts
    export FONTCONFIG_FILE=/etc/fonts/fonts.conf
    
    echo "==================================================="
    echo "  Data Science Environment Ready"
    echo "  - Jupyter: run 'jupyter lab'"
    echo "  - VS Code: run 'code .'"
    echo "==================================================="
  '';
}
