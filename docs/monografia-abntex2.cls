%% Customizações do abnTeX2 (http://abnTeX2.googlecode.com) para a PPGEC UPE
%%
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either version 1.3
%% of this license or (at your option) any later version.
%% The latest version of this license is in
%%   http://www.latex-project.org/lppl.txt
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 2005/12/01 or later.
%%
%% This work has the LPPL maintenance status `maintained'.
%%
%% The Current Maintainer of this work is SEU_NOME, SEU_EMAIL
%%
%% Further information about abnTeX2 are available on http://abntex2.googlecode.com/
%%


\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ppgec-abntex2}[01/09/2016/ v1.0 Modelo do PPGEC-UPE baseado no abnTeX2]

%
% Repassa todos os parametros a classe abntex2,
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{abntex2}%
}
\ProcessOptions\relax
\LoadClass{abntex2}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FONTES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Redefinição de fontes
\renewcommand{\ABNTEXchapterfont}{\bfseries}

% Estilo de capítulo conforme o padrão abnTeX2
\chapterstyle{abnt}

% Faz com que as notas de rodapé parem de resetar após novos capítulos
\counterwithout{footnote}{chapter}	

% Redefinição de nome de elementos pré-textuais
\renewcommand{\listadesiglasname}{\centering{Lista de abreviaturas e siglas}}

\addto\captionsbrazil{% portugues-brasil
  %% ajusta nomes padroes do babel
  \renewcommand{\bibname}{\centering{Refer\^encias}}
  \renewcommand{\contentsname}{\centering{Sum\'ario}}
  \renewcommand{\listfigurename}{\centering{Lista de ilustrações}}
  \renewcommand{\listtablename}{\centering{Lista de tabelas}}
  %% ajusta nomes usados com a macro \autoref
  \renewcommand{\pageautorefname}{p\'agina}
  \renewcommand{\sectionautorefname}{se{\c c}\~ao}
  \renewcommand{\subsectionautorefname}{subse{\c c}\~ao}
  \renewcommand{\subsubsectionautorefname}{subse{\c c}\~ao}
  \renewcommand{\paragraphautorefname}{subse{\c c}\~ao}
}


%%% ATENÇÃO!!! %%%
%%% Preencha estes comandos com suas informações
\newcommand{\logo}{\includegraphics[scale=0.67]{figuras/brasao}} % inclua o arquivo com o logo da instituição
\newcommand{\logoesq}{\includegraphics[scale=0.15]{imagens/logo-poli}}
\newcommand{\logodir}{\includegraphics[scale=0.20]{imagens/logo-ecomp}}

\newcommand{\cabcentroinicio}{{\large \univ} \\
{\large \escola} \\ {\large \disc} \\ \tit\, -\, \data}



%%%% alterando a capa %%%%
\renewcommand{\imprimircapa}{%
  \begin{capa}%
	\vspace*{-1cm}
  	\hspace{-1cm}
	\begin{minipage}{3cm}
	\logoesq
	\end{minipage}
	\hspace{8cm}
	\begin{minipage}{3cm}
	\logodir
	\end{minipage}
    \center
    
    % \includegraphics[scale=0.3]{imagens/brasao}
    \vfill
    
    \begin{SingleSpace} \Huge\textbf\imprimirtitulo \end{SingleSpace}
	\vfill
	
	{\Large\textsc\imprimirtipotrabalho}
	
    % {\SingleSpacing \large \imprimirinstituicao}

    \vfill

    {\large\textsc\imprimirautor} \\
    \vspace{1mm}
    {\large\textsc{Orientador: \imprimirorientador}} \\
    

    \vfill


	\includegraphics[scale=0.3]{imagens/brasao}


    \vfill
  \end{capa}
}


%%%% folha de rosto %%%%

\makeatletter

\renewcommand{\folhaderostocontent}{
\begin{folhaderosto}
	
	\vspace*{-1cm}
  	\hspace{-1cm}
	\begin{minipage}{3cm}
	\logoesq
	\end{minipage}
	\hspace{8cm}
	\begin{minipage}{3cm}
	\logodir
	\end{minipage}
    \center


    {\SingleSpacing \Large \imprimirinstituicao}

{
\vfill
\huge\imprimirautor
\vfill

\begin{SingleSpace} \Huge\textbf\imprimirtitulo \end{SingleSpace}

\vfill


\hspace{.45\textwidth}
\begin{minipage}{.5\textwidth}
\normalsize\imprimirpreambulo

\vfill

\end{minipage}

\vspace{20mm}

    {\normalsize\text{Orientador: \imprimirorientador}} \\
    \vspace{-7mm}
    


%\end{flushright}


\vfill

}



\normalsize\imprimirlocal, \imprimirdata

% \vspace*{2.2cm}

\end{folhaderosto}
}


\makeatother
